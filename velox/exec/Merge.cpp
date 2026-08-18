/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/exec/Merge.h"
#include <algorithm>
#include <folly/Traits.h>
#include <exception>
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/vector/DecodedVector.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::exec {

Merge::Merge(
    int32_t operatorId,
    DriverCtx* driverCtx,
    RowTypePtr outputType,
    const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
        sortingKeys,
    const std::vector<core::SortOrder>& sortingOrders,
    const std::string& planNodeId,
    std::string_view operatorType,
    const std::optional<common::SpillConfig>& spillConfig)
    : SourceOperator(
          driverCtx,
          std::move(outputType),
          operatorId,
          planNodeId,
          operatorType,
          spillConfig),
      maxOutputBatchRows_{outputBatchRows()},
      maxOutputBatchBytes_{
          driverCtx->queryConfig().preferredOutputBatchBytes()},
      sortingKeys_([&]() {
        auto numKeys = sortingKeys.size();
        std::vector<SpillSortKey> keys;
        keys.reserve(numKeys);
        for (int i = 0; i < numKeys; ++i) {
          auto channel = exprToChannel(sortingKeys[i].get(), outputType_);
          VELOX_CHECK_NE(
              channel,
              kConstantChannel,
              "Merge doesn't allow constant grouping keys");
          keys.emplace_back(
              channel,
              CompareFlags{
                  sortingOrders[i].isNullsFirst(),
                  sortingOrders[i].isAscending(),
                  false});
        }
        return keys;
      }()) {}

void Merge::initialize() {
  Operator::initialize();
  VELOX_CHECK_EQ(mergeStats_.streamingSourceReadStartTimeUs, 0);
  mergeStats_.streamingSourceReadStartTimeUs = getCurrentTimeMicro();
}

BlockingReason Merge::isBlocked(ContinueFuture* future) {
  TestValue::adjust("facebook::velox::exec::Merge::isBlocked", this);

  const auto reason = addMergeSources(future);
  if (reason != BlockingReason::kNotBlocked) {
    return reason;
  }

  // NOTE: the task might terminate early which leaves empty sources. Once it
  // happens, we shall simply mark the merge operator as finished.
  if (sources_.empty()) {
    finished_ = true;
    return BlockingReason::kNotBlocked;
  }

  maybeStartNextMergeSourceGroup();

  if (sourceMerger_ != nullptr) {
    sourceMerger_->isBlocked(sourceBlockingFutures_);
  }

  if (sourceBlockingFutures_.empty()) {
    return BlockingReason::kNotBlocked;
  }

  // Wait for any source to become available instead of one specific source:
  // with range-partitioned sources a source can stay empty until its producer
  // finishes, and blocking on it would stall the merge even when other
  // sources already have data (the producer is then blocked on the merge's
  // backpressure, a cycle). Waiting on any source keeps progress.
  *future = folly::collectAny(sourceBlockingFutures_).unit();
  sourceBlockingFutures_.clear();
  return BlockingReason::kWaitForProducer;
}

bool Merge::isFinished() {
  return finished_;
}

void Merge::maybeSetupOutputSpiller() {
  VELOX_CHECK(canSpill());
  VELOX_CHECK(spillConfig_.has_value());
  if (mergeOutputSpiller_ != nullptr) {
    return;
  }

  mergeOutputSpiller_ = std::make_unique<MergeSpiller>(
      outputType_,
      std::nullopt,
      HashBitRange{},
      sortingKeys_,
      &spillConfig_.value(),
      spillStats_.get());
}

void Merge::spill() {
  if (output_ == nullptr) {
    return;
  }
  maybeSetupOutputSpiller();
  numSpilledRows_ += output_->size();
  mergeOutputSpiller_->spill(SpillPartitionId{0}, output_);
  output_ = nullptr;
}

void Merge::finishMergeSourceGroup() {
  sourceMerger_ = nullptr;
  if (mergeOutputSpiller_ == nullptr) {
    return;
  }
  VELOX_CHECK(needSpill());
  VELOX_CHECK_GT(numSpilledRows_, 0);
  // Finishes spill if it has happened and setup spill merger if no more source
  // to merge.
  SpillPartitionSet spillPartitionSet;
  mergeOutputSpiller_->finishSpill(spillPartitionSet);
  mergeOutputSpiller_ = nullptr;
  VELOX_CHECK_EQ(spillPartitionSet.size(), 1);
  auto spillFiles = spillPartitionSet.begin()->second->files();
  VELOX_CHECK(!spillFiles.empty());
  spillFileGroups_.push_back(std::move(spillFiles));
}

void Merge::setupSpillMerger() {
  VELOX_CHECK(!spillFileGroups_.empty());
  VELOX_CHECK_NULL(spillMerger_);
  VELOX_CHECK(spillConfig_.has_value());
  std::vector<std::vector<std::unique_ptr<SpillReadFile>>> spillReadFilesGroups;
  spillReadFilesGroups.reserve(spillFileGroups_.size());
  for (const auto& spillFiles : spillFileGroups_) {
    std::vector<std::unique_ptr<SpillReadFile>> spillReadFiles;
    spillReadFiles.reserve(spillFiles.size());
    for (const auto& spillFile : spillFiles) {
      spillReadFiles.emplace_back(
          SpillReadFile::create(
              spillFile,
              spillConfig_->readBufferSize,
              pool(),
              spillStats_.get()));
    }
    spillReadFilesGroups.push_back(std::move(spillReadFiles));
  }
  spillFileGroups_.clear();
  spillMerger_ = std::make_shared<SpillMerger>(
      sortingKeys_,
      outputType_,
      std::move(spillReadFilesGroups),
      maxOutputBatchRows_,
      maxOutputBatchBytes_,
      operatorCtx_->driverCtx()->queryConfig().localMergeSourceQueueSize(),
      &spillConfig_.value(),
      spillStats_,
      pool());
  spillMerger_->start();
}

void Merge::maybeStartNextMergeSourceGroup() {
  if (sourceMerger_ != nullptr || numStartedSources_ >= sources_.size()) {
    return;
  }

  // Gets the merge sources for the next partial merge run.
  std::vector<MergeSource*> sources;
  for (auto i = numStartedSources_; i <
       (std::min(sources_.size(), numStartedSources_ + maxNumMergeSources_));
       ++i) {
    sources.push_back(sources_[i].get());
  }

  // Initializes the source merger.
  std::vector<std::unique_ptr<SourceStream>> cursors;
  cursors.reserve(sources.size());
  for (auto* source : sources) {
    cursors.push_back(
        std::make_unique<SourceStream>(
            source, sortingKeys_, maxOutputBatchRows_));
  }

  // TODO: consider to provide a config other than the regular operator batch
  // size to tune the batch size of the streaming source merge output as the
  // merge operator is single threaded.
  sourceMerger_ = std::make_unique<SourceMerger>(
      outputType_,
      std::move(cursors),
      maxOutputBatchRows_,
      maxOutputBatchBytes_,
      pool());
  // Start sources.
  for (const auto& source : sources) {
    source->start();
  }
  numStartedSources_ += sources.size();
}

RowVectorPtr Merge::getOutputFromSpill() {
  VELOX_CHECK_NOT_NULL(spillMerger_);
  VELOX_CHECK_NULL(sourceMerger_);
  bool atEnd = false;
  output_ = spillMerger_->getOutput(sourceBlockingFutures_, atEnd);
  SCOPE_EXIT {
    if (!atEnd) {
      return;
    }
    finished_ = true;
    VELOX_CHECK_EQ(mergeStats_.spilledSourceReadEndTimeUs, 0);
    mergeStats_.spilledSourceReadEndTimeUs = getCurrentTimeMicro();
  };
  return std::move(output_);
}

RowVectorPtr Merge::getOutputFromSource() {
  VELOX_CHECK_NULL(spillMerger_);
  bool atEnd = false;
  output_ = sourceMerger_->getOutput(sourceBlockingFutures_, atEnd);
  if (needSpill()) {
    spill();
    VELOX_CHECK_NULL(output_);
  }

  if (!atEnd) {
    return std::move(output_);
  }

  finishMergeSourceGroup();
  if (numStartedSources_ < sources_.size()) {
    VELOX_CHECK_NULL(output_);
    return nullptr;
  }

  VELOX_CHECK_EQ(mergeStats_.streamingSourceReadEndTimeUs, 0);
  mergeStats_.streamingSourceReadEndTimeUs = getCurrentTimeMicro();

  if (numSpilledRows_ > 0) {
    setupSpillMerger();
    VELOX_CHECK_NULL(output_);
    return nullptr;
  }

  finished_ = true;
  return std::move(output_);
}

RowVectorPtr Merge::getOutput() {
  if (finished_) {
    return nullptr;
  }

  // Read from spill.
  if (spillMerger_ != nullptr) {
    return getOutputFromSpill();
  }

  return getOutputFromSource();
}

void Merge::close() {
  recordMergeStats();
  for (auto& source : sources_) {
    source->close();
  }
  Operator::close();
}

void Merge::recordMergeStats() {
  auto lockedStats = stats_.wlock();
  if (mergeStats_.streamingSourceReadEndTimeUs > 0) {
    VELOX_CHECK_GT(mergeStats_.streamingSourceReadStartTimeUs, 0);
    VELOX_CHECK_GE(
        mergeStats_.streamingSourceReadEndTimeUs,
        mergeStats_.streamingSourceReadStartTimeUs);
    lockedStats->addRuntimeStat(
        kStreamingSourceReadWallNanos,
        RuntimeCounter(
            (mergeStats_.streamingSourceReadEndTimeUs -
             mergeStats_.streamingSourceReadStartTimeUs) *
                1'000,
            RuntimeCounter::Unit::kNanos));
  }
  if (mergeStats_.spilledSourceReadEndTimeUs > 0) {
    VELOX_CHECK_GT(mergeStats_.streamingSourceReadEndTimeUs, 0);
    VELOX_CHECK_GE(
        mergeStats_.spilledSourceReadEndTimeUs,
        mergeStats_.streamingSourceReadEndTimeUs);
    VELOX_CHECK_GT(numSpilledRows_, 0);
    lockedStats->addRuntimeStat(
        kSpilledSourceReadWallNanos,
        RuntimeCounter(
            (mergeStats_.spilledSourceReadEndTimeUs -
             mergeStats_.streamingSourceReadEndTimeUs) *
                1'000,
            RuntimeCounter::Unit::kNanos));
  }
}

SourceMerger::SourceMerger(
    const RowTypePtr& type,
    std::vector<std::unique_ptr<SourceStream>> sourceStreams,
    vector_size_t maxOutputBatchRows,
    uint64_t maxOutputBatchBytes,
    velox::memory::MemoryPool* pool)
    : type_(type),
      maxOutputBatchRows_(maxOutputBatchRows),
      maxOutputBatchBytes_(maxOutputBatchBytes),
      streams_([&sourceStreams]() {
        std::vector<SourceStream*> streams;
        for (auto& cursor : sourceStreams) {
          streams.push_back(cursor.get());
        }
        return streams;
      }()),
      merger_(
          std::make_unique<TreeOfLosers<SourceStream>>(
              std::move(sourceStreams))),
      pool_(pool) {}

void SourceMerger::isBlocked(
    std::vector<ContinueFuture>& sourceBlockingFutures) const {
  if (sourceBlockingFutures.empty()) {
    for (auto* stream : streams_) {
      stream->isBlocked(sourceBlockingFutures);
    }
  }
}

void SourceMerger::setOutputBatchSize() {
  if (outputBatchRows_ != 0) {
    return;
  }
  size_t numEstimations{0};
  int64_t estimateRowSizeSum{0};
  for (auto* stream : streams_) {
    const auto estimateRowSize = stream->estimateRowSize();
    if (estimateRowSize.has_value()) {
      ++numEstimations;
      estimateRowSizeSum += estimateRowSize.value();
    }
  }

  if (numEstimations == 0) {
    outputBatchRows_ = maxOutputBatchRows_;
    return;
  }

  const auto estimateRowSize =
      std::max<vector_size_t>(1, estimateRowSizeSum / numEstimations);
  outputBatchRows_ = std::min<vector_size_t>(
      std::max<vector_size_t>(1, maxOutputBatchBytes_ / estimateRowSize),
      maxOutputBatchRows_);
}

RowVectorPtr SourceMerger::getOutput(
    std::vector<ContinueFuture>& sourceBlockingFutures,
    bool& atEnd) {
  VELOX_CHECK_NOT_NULL(merger_);
  atEnd = false;
  setOutputBatchSize();
  VELOX_CHECK_GT(outputBatchRows_, 0);

  if (!output_) {
    output_ = createOutputVector();
  }

  for (;;) {
    auto stream = merger_->next();

    if (!stream) {
      atEnd = true;

      // Return nullptr if there is no data.
      if (outputRows_ == 0) {
        return nullptr;
      }
      output_->resize(outputRows_);
      return std::move(output_);
    }

    if (stream->setOutputRow(outputRows_)) {
      // The stream is at end of input batch. Need to copy out the rows before
      // fetching next batch in 'pop'.
      stream->copyToOutput(output_);
      TestValue::adjust(
          "facebook::velox::exec::SourceMerger::getOutput",
          &sourceBlockingFutures);
    }

    ++outputRows_;

    // Advance the stream.
    stream->pop(sourceBlockingFutures);

    if (outputRows_ == outputBatchRows_) {
      // Copy out data from all sources.
      for (auto& s : streams_) {
        s->copyToOutput(output_);
      }
      outputRows_ = 0;
      return std::move(output_);
    }

    if (!sourceBlockingFutures.empty()) {
      return nullptr;
    }
  }
}

RowVectorPtr SourceMerger::createOutputVector() {
  // Attempt to generate output vector using stream data to preserve encodings.
  // First, find the first stream with non-null data to determine column
  // encodings.
  const RowVector* source = nullptr;
  for (const auto* stream : streams_) {
    if (stream->hasData() && (source = stream->data())) {
      return BaseVector::createEmptyLike<RowVector>(
          source, outputBatchRows_, pool_);
    }
  }

  // If a non-null stream cannot be found, default to generating row vector by
  // type.
  return BaseVector::create<RowVector>(type_, outputBatchRows_, pool_);
}

bool SourceStream::operator<(const MergeStream& other) const {
  const auto& otherCursor = static_cast<const SourceStream&>(other);
  for (auto i = 0; i < sortingKeys_.size(); ++i) {
    const auto& [_, compareFlags] = sortingKeys_[i];
    VELOX_DCHECK(
        compareFlags.nullAsValue(), "not supported null handling mode");
    if (auto result = keyColumns_[i]
                          ->compare(
                              otherCursor.keyColumns_[i],
                              currentSourceRow_,
                              otherCursor.currentSourceRow_,
                              compareFlags)
                          .value()) {
      return result < 0;
    }
  }
  return false;
}

bool SourceStream::pop(std::vector<ContinueFuture>& futures) {
  ++currentSourceRow_;
  if (currentSourceRow_ == data_->size()) {
    // Make sure all current data has been copied out.
    VELOX_CHECK(!outputRows_.hasSelections());
    return fetchMoreData(futures);
  }

  return false;
}

void SourceStream::copyToOutput(RowVectorPtr& output) {
  outputRows_.updateBounds();

  if (!outputRows_.hasSelections()) {
    return;
  }

  vector_size_t sourceRow = firstSourceRow_;
  outputRows_.applyToSelected(
      [&](auto row) { sourceRows_[row] = sourceRow++; });

  for (auto i = 0; i < output->type()->size(); ++i) {
    output->childAt(i)->copy(
        data_->childAt(i).get(), outputRows_, sourceRows_.data());
  }

  outputRows_.clearAll();

  if (sourceRow == data_->size()) {
    firstSourceRow_ = 0;
  } else {
    firstSourceRow_ = sourceRow;
  }
}

bool SourceStream::fetchMoreData(std::vector<ContinueFuture>& futures) {
  ContinueFuture future;
  bool drained{false};
  auto reason = source_->next(data_, &future, drained);
  if (reason != BlockingReason::kNotBlocked) {
    needData_ = true;
    futures.emplace_back(std::move(future));
    return true;
  }

  atEnd_ = !data_ || data_->size() == 0;
  needData_ = false;
  currentSourceRow_ = 0;

  if (!atEnd_) {
    for (auto& child : data_->children()) {
      child = BaseVector::loadedVectorShared(child);
    }
    keyColumns_.clear();
    for (const auto& key : sortingKeys_) {
      keyColumns_.push_back(data_->childAt(key.first).get());
    }
  }
  return false;
}

SpillMerger::SpillMerger(
    const std::vector<SpillSortKey>& sortingKeys,
    const RowTypePtr& type,
    std::vector<std::vector<std::unique_ptr<SpillReadFile>>>
        spillReadFilesGroup,
    vector_size_t maxOutputBatchRows,
    uint64_t maxOutputBatchBytes,
    int mergeSourceQueueSize,
    const common::SpillConfig* spillConfig,
    const std::shared_ptr<exec::SpillStats>& spillStats,
    velox::memory::MemoryPool* pool)
    : executor_(spillConfig->executor),
      spillStats_(spillStats),
      pool_(pool->shared_from_this()),
      sources_(
          createMergeSources(spillReadFilesGroup.size(), mergeSourceQueueSize)),
      batchStreams_(createBatchStreams(std::move(spillReadFilesGroup))),
      // TODO: consider to provide a config other than the regular operator
      // batch size to tune the batch size of the spilled source merge output as
      // the merge operator is single threaded.
      sourceMerger_(createSourceMerger(
          sortingKeys,
          type,
          sources_,
          maxOutputBatchRows,
          maxOutputBatchBytes,
          pool)) {}

SpillMerger::~SpillMerger() {
  sourceMerger_.reset();
  batchStreams_.clear();
  sources_.clear();
}

void SpillMerger::start() {
  VELOX_CHECK_NOT_NULL(
      executor_,
      "SpillMerge require configure executor to run async spill file stream producer");
  scheduleAsyncSpillFileStreamReads();
}

RowVectorPtr SpillMerger::getOutput(
    std::vector<ContinueFuture>& sourceBlockingFutures,
    bool& atEnd) {
  TestValue::adjust(
      "facebook::velox::exec::SpillMerger::getOutput", &sourceBlockingFutures);
  sourceMerger_->isBlocked(sourceBlockingFutures);
  if (!sourceBlockingFutures.empty()) {
    return nullptr;
  }
  // SpillMerger::getOutput waits for all readers to finish, reaches EOF,
  // and rethrows any captured error. Centralizing error propagation here
  // helps prevent potential resource leaks.
  auto output = sourceMerger_->getOutput(sourceBlockingFutures, atEnd);
  if (atEnd) {
    checkError();
  }
  return output;
}

std::vector<std::shared_ptr<MergeSource>> SpillMerger::createMergeSources(
    size_t numSpillSources,
    int queueSize) {
  std::vector<std::shared_ptr<MergeSource>> sources;
  sources.reserve(numSpillSources);
  for (auto i = 0; i < numSpillSources; ++i) {
    sources.push_back(MergeSource::createLocalMergeSource(queueSize));
  }
  for (const auto& source : sources) {
    source->start();
  }
  return sources;
}

std::vector<std::unique_ptr<BatchStream>> SpillMerger::createBatchStreams(
    std::vector<std::vector<std::unique_ptr<SpillReadFile>>>
        spillReadFilesGroup) {
  const auto numStreams = spillReadFilesGroup.size();
  std::vector<std::unique_ptr<BatchStream>> batchStreams;
  batchStreams.reserve(numStreams);
  for (auto i = 0; i < numStreams; ++i) {
    batchStreams.emplace_back(
        ConcatFilesSpillBatchStream::create(std::move(spillReadFilesGroup[i])));
  }
  return batchStreams;
}

std::unique_ptr<SourceMerger> SpillMerger::createSourceMerger(
    const std::vector<SpillSortKey>& sortingKeys,
    const RowTypePtr& type,
    const std::vector<std::shared_ptr<MergeSource>>& sources,
    vector_size_t maxOutputBatchRows,
    uint64_t maxOutputBatchBytes,
    velox::memory::MemoryPool* pool) {
  std::vector<std::unique_ptr<SourceStream>> streams;
  streams.reserve(sources.size());
  for (const auto& source : sources) {
    streams.push_back(
        std::make_unique<SourceStream>(
            source.get(), sortingKeys, maxOutputBatchRows));
  }
  return std::make_unique<SourceMerger>(
      type, std::move(streams), maxOutputBatchRows, maxOutputBatchBytes, pool);
}

void SpillMerger::finishSource(size_t streamIdx) const {
  ContinueFuture future{ContinueFuture::makeEmpty()};
  sources_[streamIdx]->enqueue(nullptr, &future);
  VELOX_CHECK(!future.valid());
}

void SpillMerger::readFromSpillFileStream(
    const std::weak_ptr<SpillMerger>& mergeHolder,
    size_t streamIdx) {
  TestValue::adjust(
      "facebook::velox::exec::SpillMerger::readFromSpillFileStream", nullptr);
  const auto merger = mergeHolder.lock();
  if (merger == nullptr) {
    LOG(ERROR) << "SpillMerger is destroyed, abandon reading from batch stream";
    return;
  }

  try {
    if (hasError()) {
      finishSource(streamIdx);
      return;
    }

    RowVectorPtr vector;
    if (!batchStreams_[streamIdx]->nextBatch(vector)) {
      VELOX_CHECK_NULL(vector);
      finishSource(streamIdx);
      return;
    }

    ContinueFuture future{ContinueFuture::makeEmpty()};
    const auto blockingReason =
        sources_[streamIdx]->enqueue(std::move(vector), &future);
    if (blockingReason == BlockingReason::kNotBlocked) {
      VELOX_CHECK(!future.valid());
      readFromSpillFileStream(mergeHolder, streamIdx);
    } else {
      VELOX_CHECK(future.valid());
      std::move(future)
          .via(executor_)
          .thenValue([this, mergeHolder, streamIdx](auto&&) {
            readFromSpillFileStream(mergeHolder, streamIdx);
          })
          .thenError(
              folly::tag_t<std::exception>{},
              [this, mergeHolder, streamIdx](const std::exception& e) {
                const auto merger = mergeHolder.lock();
                if (merger != nullptr) {
                  LOG(ERROR) << "Stop the " << streamIdx
                             << " th source on error: " << e.what();
                  setError(std::make_exception_ptr(e));
                  finishSource(streamIdx);
                }
              });
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "The " << streamIdx
               << " spill stream failed with error: " << e.what();
    setError(std::current_exception());
    finishSource(streamIdx);
  }
}

void SpillMerger::scheduleAsyncSpillFileStreamReads() {
  VELOX_CHECK_EQ(batchStreams_.size(), sources_.size());
  for (auto i = 0; i < batchStreams_.size(); ++i) {
    executor_->add([&, streamIdx = i]() {
      readFromSpillFileStream(std::weak_ptr(shared_from_this()), streamIdx);
    });
  }
}

void SpillMerger::setError(const std::exception_ptr& exception) {
  std::lock_guard l(mutex_);
  if (exception_ != nullptr) {
    return;
  }
  exception_ = exception;
}

bool SpillMerger::hasError() const {
  std::lock_guard l(mutex_);
  return exception_ != nullptr;
}

void SpillMerger::checkError() {
  if (hasError()) {
    sourceMerger_.reset();
    batchStreams_.clear();
    sources_.clear();
    std::rethrow_exception(exception_);
  }
}

LocalMerge::LocalMerge(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::LocalMergeNode>& localMergeNode)
    : Merge(
          operatorId,
          driverCtx,
          localMergeNode->outputType(),
          localMergeNode->sortingKeys(),
          localMergeNode->sortingOrders(),
          localMergeNode->id(),
          OperatorType::kLocalMerge,
          localMergeNode->canSpill(driverCtx->queryConfig())
              ? driverCtx->makeSpillConfig(
                    operatorId,
                    OperatorType::kLocalMerge)
              : std::nullopt),
      partitionId_(driverCtx->partitionId),
      rangePartitioned_(localMergeNode->rangePartitionSpec().has_value()) {
  if (!rangePartitioned_) {
    VELOX_CHECK_EQ(
        operatorCtx_->driverCtx()->driverId,
        0,
        "LocalMerge needs to run single-threaded");
  }
  // Enable local merge spill iff spill is enabled and the spill executor is
  // provided.
  if (spillConfig_.has_value() && spillConfig_->executor != nullptr) {
    maxNumMergeSources_ = operatorCtx_->task()
                              ->queryCtx()
                              ->queryConfig()
                              .localMergeMaxNumMergeSources();
  }
}

BlockingReason LocalMerge::addMergeSources(ContinueFuture* /* future */) {
  if (sources_.empty()) {
    auto allSources = operatorCtx_->task()->getLocalMergeSources(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());
    if (rangePartitioned_) {
      // Each driver merges only the merge sources of its own range bucket.
      for (const auto& source : allSources) {
        if (source->partitionId() == partitionId_) {
          sources_.push_back(source);
        }
      }
    } else {
      sources_ = std::move(allSources);
    }
  }
  return BlockingReason::kNotBlocked;
}

MergeExchange::MergeExchange(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::MergeExchangeNode>& mergeExchangeNode)
    : Merge(
          operatorId,
          driverCtx,
          mergeExchangeNode->outputType(),
          mergeExchangeNode->sortingKeys(),
          mergeExchangeNode->sortingOrders(),
          mergeExchangeNode->id(),
          OperatorType::kMergeExchange),
      serde_(getNamedVectorSerde(mergeExchangeNode->serdeKind())),
      serdeOptions_(getVectorSerdeOptions(
          common::stringToCompressionKind(
              driverCtx->queryConfig().shuffleCompressionKind()),
          mergeExchangeNode->serdeKind(),
          std::nullopt,
          driverCtx->queryConfig().minShuffleCompressionPageSizeBytes())) {}

BlockingReason MergeExchange::addMergeSources(ContinueFuture* future) {
  if (operatorCtx_->driverCtx()->driverId != 0) {
    // When there are multiple pipelines, a single operator, the one from
    // pipeline 0, is responsible for merging pages.
    return BlockingReason::kNotBlocked;
  }
  if (noMoreSplits_) {
    return BlockingReason::kNotBlocked;
  }

  for (;;) {
    exec::Split split;
    auto reason = operatorCtx_->task()->getSplitOrFuture(

        operatorCtx_->driverCtx()->driverId,
        operatorCtx_->driverCtx()->splitGroupId,
        planNodeId(),
        /*maxPreloadSplits=*/0,
        /*preload=*/nullptr,
        split,
        *future);
    if (reason != BlockingReason::kNotBlocked) {
      return reason;
    }

    if (split.hasConnectorSplit()) {
      auto remoteSplit =
          std::dynamic_pointer_cast<RemoteConnectorSplit>(split.connectorSplit);
      VELOX_CHECK_NOT_NULL(remoteSplit, "Wrong type of split");
      remoteSourceTaskIds_.push_back(remoteSplit->taskId);
      continue;
    }

    noMoreSplits_ = true;
    if (!remoteSourceTaskIds_.empty()) {
      const auto maxMergeExchangeBufferSize =
          operatorCtx_->driverCtx()->queryConfig().maxMergeExchangeBufferSize();
      const auto maxQueuedBytesPerSource = std::min<int64_t>(
          std::max<int64_t>(
              maxMergeExchangeBufferSize / remoteSourceTaskIds_.size(),
              MergeSource::kMaxQueuedBytesLowerLimit),
          MergeSource::kMaxQueuedBytesUpperLimit);
      for (uint32_t remoteSourceIndex = 0;
           remoteSourceIndex < remoteSourceTaskIds_.size();
           ++remoteSourceIndex) {
        auto* pool = operatorCtx_->task()->addMergeSourcePool(
            operatorCtx_->planNodeId(),
            operatorCtx_->driverCtx()->pipelineId,
            remoteSourceIndex);
        sources_.emplace_back(
            MergeSource::createMergeExchangeSource(
                this,
                remoteSourceTaskIds_[remoteSourceIndex],
                operatorCtx_->task()->destination(),
                maxQueuedBytesPerSource,
                pool,
                operatorCtx_->task()->queryCtx()->executor()));
      }
    }
    // TODO Delay this call until all input data has been processed.
    operatorCtx_->task()->multipleSplitsFinished(
        false, remoteSourceTaskIds_.size(), 0);
    return BlockingReason::kNotBlocked;
  }
}

void MergeExchange::close() {
  for (auto& source : sources_) {
    source->close();
  }
  Operator::close();
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        Operator::kShuffleSerdeKind,
        RuntimeCounter(
            static_cast<int64_t>(VectorSerde::kindByName(serde_->kind()))));
    lockedStats->addRuntimeStat(
        Operator::kShuffleCompressionKind,
        RuntimeCounter(static_cast<int64_t>(serdeOptions_->compressionKind)));
  }
}

RangePartitionedMergeSink::RangePartitionedMergeSink(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::LocalMergeNode>& localMergeNode)
    : Operator(
          driverCtx,
          localMergeNode->outputType(),
          operatorId,
          localMergeNode->id(),
          OperatorType::kRangePartitionedMerge),
      numPartitions_(localMergeNode->rangePartitionSpec()->numPartitions),
      boundaries_(localMergeNode->rangePartitionSpec()->boundaries),
      nullsFirst_(localMergeNode->sortingOrders()[0].isNullsFirst()),
      keyChannel_(
          exprToChannel(
              localMergeNode->sortingKeys()[0].get(),
              localMergeNode->outputType())),
      partitionEnded_(numPartitions_, false) {
  VELOX_CHECK_GT(numPartitions_, 1);
  VELOX_CHECK_EQ(boundaries_.size(), numPartitions_ - 1);
  for (auto i = 0; i < numPartitions_; ++i) {
    sources_.push_back(operatorCtx_->task()->addLocalMergeSource(
        operatorCtx_->driverCtx()->splitGroupId,
        planNodeId(),
        localMergeNode->outputType(),
        operatorCtx_->driverCtx()->queryConfig().localMergeSourceQueueSize(),
        i));
  }
}

namespace {

// The bucket of 'value' under the ascending boundaries: rows with
// key < boundaries[j] belong to bucket j.
template <typename T>
int32_t bucketForValue(
    const std::vector<facebook::velox::variant>& boundaries,
    T value) {
  int32_t lo = 0;
  int32_t hi = boundaries.size();
  while (lo < hi) {
    const int32_t mid = (lo + hi) / 2;
    if (value < boundaries[mid].value<T>()) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

} // namespace

void RangePartitionedMergeSink::addInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }

  auto keyVector = input->childAt(keyChannel_);
  keyVector->loadedVector();
  DecodedVector decoded(*keyVector);
  const TypeKind keyKind = keyVector->typeKind();

  std::vector<vector_size_t> counts(numPartitions_, 0);
  std::vector<vector_size_t> partitions(input->size());
  for (auto row = 0; row < input->size(); ++row) {
    int32_t partition;
    if (decoded.isNullAt(row)) {
      partition = nullsFirst_ ? 0 : numPartitions_ - 1;
    } else {
      switch (keyKind) {
        case TypeKind::BIGINT:
          partition = bucketForValue(
              boundaries_, decoded.valueAt<int64_t>(row));
          break;
        case TypeKind::INTEGER:
          partition = bucketForValue(
              boundaries_, decoded.valueAt<int32_t>(row));
          break;
        case TypeKind::SMALLINT:
          partition = bucketForValue(
              boundaries_, decoded.valueAt<int16_t>(row));
          break;
        case TypeKind::VARCHAR:
          partition = bucketForValue(
              boundaries_,
              std::string(decoded.valueAt<StringView>(row)));
          break;
        default:
          VELOX_UNSUPPORTED(
              "RangePartitionedMergeSink does not support key type {}",
              TypeKindName::toName(keyKind));
      }
    }
    VELOX_DCHECK_GE(partition, 0);
    VELOX_DCHECK_LT(partition, numPartitions_);
    partitions[row] = partition;
    ++counts[partition];
  }

  // Signal end-of-data to buckets that this sorted run has moved past: the
  // row partitions are non-decreasing, so once the current batch's lowest
  // bucket is above a bucket, no later row will ever be routed to it. Doing
  // this early lets the bucket's merge produce without waiting for this sink
  // to fully finish (which could otherwise be blocked on the merge's
  // backpressure, a cycle).
  int32_t minPartition = numPartitions_;
  for (auto row = 0; row < input->size(); ++row) {
    minPartition = std::min(minPartition, partitions[row]);
  }
  for (auto partition = 0; partition < minPartition; ++partition) {
    if (!partitionEnded_[partition]) {
      ContinueFuture future;
      const auto reason = sources_[partition]->enqueue(nullptr, &future);
      VELOX_CHECK_EQ(reason, BlockingReason::kNotBlocked);
      partitionEnded_[partition] = true;
    }
  }

  std::vector<std::vector<vector_size_t>> indices(numPartitions_);
  for (auto partition = 0; partition < numPartitions_; ++partition) {
    indices[partition].reserve(counts[partition]);
  }
  for (auto row = 0; row < input->size(); ++row) {
    indices[partitions[row]].push_back(row);
  }

  for (auto partition = 0; partition < numPartitions_; ++partition) {
    if (indices[partition].empty()) {
      continue;
    }
    auto mapping = allocateIndices(indices[partition].size(), pool());
    auto* rawMapping = mapping->asMutable<vector_size_t>();
    memcpy(
        rawMapping,
        indices[partition].data(),
        indices[partition].size() * sizeof(vector_size_t));
    auto partitionData = exec::wrap(
        indices[partition].size(), mapping, input);
    enqueueRowVector(partition, std::move(partitionData));
  }
}

void RangePartitionedMergeSink::enqueueRowVector(
    int32_t partition,
    RowVectorPtr vector) {
  ContinueFuture future;
  auto reason = sources_[partition]->enqueue(std::move(vector), &future);
  if (reason != BlockingReason::kNotBlocked) {
    blockingFutures_.push_back(std::move(future));
  }
}

void RangePartitionedMergeSink::noMoreInput() {
  Operator::noMoreInput();
  for (auto partition = 0; partition < numPartitions_; ++partition) {
    if (partitionEnded_[partition]) {
      continue;
    }
    ContinueFuture future;
    const auto reason = sources_[partition]->enqueue(nullptr, &future);
    VELOX_CHECK_EQ(reason, BlockingReason::kNotBlocked);
    partitionEnded_[partition] = true;
  }
  finished_ = true;
}

BlockingReason RangePartitionedMergeSink::isBlocked(ContinueFuture* future) {
  if (!started_) {
    // Wait until every bucket's merge driver has started consuming, otherwise
    // enqueue() would fail its started check.
    for (auto& source : sources_) {
      if (const auto reason = source->started(future);
          reason != BlockingReason::kNotBlocked) {
        return reason;
      }
    }
    started_ = true;
  }
  if (!blockingFutures_.empty()) {
    *future =
        folly::collectAll(blockingFutures_.begin(), blockingFutures_.end())
            .unit();
    blockingFutures_.clear();
    return BlockingReason::kWaitForConsumer;
  }
  return BlockingReason::kNotBlocked;
}

OrderedConcat::OrderedConcat(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::OrderedConcatNode>& concatNode)
    : SourceOperator(
          driverCtx,
          concatNode->outputType(),
          operatorId,
          concatNode->id(),
          OperatorType::kOrderedConcat) {}

bool OrderedConcat::addMergeSources(ContinueFuture* future) {
  if (allSourcesAdded_) {
    return true;
  }
  allSourcesAdded_ = true;
  auto allSources = operatorCtx_->task()->getLocalMergeSources(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  for (const auto& source : allSources) {
    sources_.push_back(source);
  }
  // Drain the buckets in partition order: the buckets are disjoint ordered
  // ranges, so the concatenation is the globally sorted stream.
  std::sort(
      sources_.begin(),
      sources_.end(),
      [](const auto& a, const auto& b) {
        return a->partitionId() < b->partitionId();
      });
  // Start all bucket sources eagerly: each bucket's merge driver is unblocked
  // by its sink's start signal and can then start its own input sources. A
  // lazy per-bucket start would deadlock, because the range-partitioned
  // merge's input sinks wait for every bucket's sources to be started.
  for (auto& source : sources_) {
    source->start();
  }
  return true;
}

BlockingReason OrderedConcat::isBlocked(ContinueFuture* future) {
  addMergeSources(future);
  if (sources_.empty()) {
    finished_ = true;
    return BlockingReason::kNotBlocked;
  }
  if (blockedFuture_.has_value()) {
    *future = std::move(blockedFuture_.value());
    blockedFuture_.reset();
    return BlockingReason::kWaitForProducer;
  }
  return BlockingReason::kNotBlocked;
}

RowVectorPtr OrderedConcat::getOutput() {
  addMergeSources(nullptr);
  while (currentSource_ < sources_.size()) {
    auto& source = sources_[currentSource_];
    ContinueFuture future;
    RowVectorPtr output;
    bool drained = false;
    const auto reason = source->next(output, &future, drained);
    if (reason != BlockingReason::kNotBlocked) {
      blockedFuture_ = std::move(future);
      return nullptr;
    }
    if (output != nullptr) {
      return output;
    }
    // The source is exhausted (or drained under a barrier); move on to the
    // next bucket.
    ++currentSource_;
  }
  finished_ = true;
  return nullptr;
}

bool OrderedConcat::isFinished() {
  return finished_;
}

void OrderedConcat::close() {
  for (auto& source : sources_) {
    source->close();
  }
  SourceOperator::close();
}
} // namespace facebook::velox::exec
