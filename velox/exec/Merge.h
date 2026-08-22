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
#pragma once

#include "velox/common/base/TreeOfLosers.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/MergeSource.h"
#include "velox/exec/Spill.h"
#include "velox/exec/Spiller.h"

namespace facebook::velox::exec {

class SourceStream;
class SourceMerger;
class SpillMerger;

/// Coordination state for a range-partitioned merge with offset skip: the
/// ordered concat drains the buckets in partition order and assigns each
/// bucket's skip budget just before draining it; the bucket merge drivers
/// wait for their budget before producing. Budgets are exact because a sorted
/// run's rows reach the buckets in ascending range order, so every bucket's
/// data is routed before any later bucket can fill and block its producer.
struct MergeSkipState {
  explicit MergeSkipState(int64_t offset) : offset(offset) {}

  std::mutex mutex;
  // Offset rows still to be discarded before the first materialized row.
  const int64_t offset;
  // Rows of already-drained buckets (skipped prefixes plus materialized rows).
  int64_t drained = 0;
  // Per-bucket skip budget; -1 until assigned by the ordered concat.
  std::unordered_map<int32_t, int64_t> skip;
  // Actual rows skipped per bucket, published by the bucket's merge driver
  // before it signals the end of the bucket. The ordered concat uses these
  // values for 'drained' accounting: a bucket may hold fewer rows than its
  // assigned budget, so the budget alone would over-count.
  std::unordered_map<int32_t, int64_t> skipped;
  // Bucket drivers wait on these; SharedPromise allows repeated waits before
  // the budget is assigned.
  std::unordered_map<int32_t, folly::SharedPromise<folly::Unit>> promises;
  // Set by the ordered concat on close: no more budgets will be granted, so
  // waiting drivers must not re-block.
  bool closed = false;
};

// Merge operator Implementation: This implementation uses priority queue
// to perform a k-way merge of its inputs. It stops merging if any one of
// its inputs is blocked.
class Merge : public SourceOperator {
 public:
  Merge(
      int32_t operatorId,
      DriverCtx* driverCtx,
      RowTypePtr outputType,
      const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
          sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders,
      const std::string& planNodeId,
      std::string_view operatorType,
      const std::optional<common::SpillConfig>& spillConfig = std::nullopt,
      std::shared_ptr<MergeSkipState> skipState = nullptr);

  void initialize() override;

  /// Returns true when the merge may start producing. For a range-partitioned
  /// merge with offset skip this waits until the ordered concat assigned this
  /// bucket's skip budget; otherwise it is a no-op.
  virtual bool waitForSkipBudget(ContinueFuture* future) {
    return true;
  }

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  RowVectorPtr getOutput() override;

  void close() override;

  const RowTypePtr& outputType() const {
    return outputType_;
  }

  /// The name of runtime stats specific to merge.
  /// The running wall time of the merge operator reading from the streaming
  /// source. If spilling is enabled for local merge, this also includes the
  /// time that writes to the spilled source.
  static constexpr std::string_view kStreamingSourceReadWallNanos{
      "streamingSourceReadWallNanos"};
  /// The running wall time of the merge operator reading from the spilled
  /// source to produce the final output. This only applies when spilling is
  /// enabled for local merge.
  static constexpr std::string_view kSpilledSourceReadWallNanos{
      "spilledSourceReadWallNanos"};

 protected:
  virtual BlockingReason addMergeSources(ContinueFuture* future) = 0;

  std::vector<std::shared_ptr<MergeSource>> sources_;
  size_t numStartedSources_{0};
  /// Maximum number of merge sources per run.
  uint32_t maxNumMergeSources_{std::numeric_limits<uint32_t>::max()};

  /// Merge offset-skip coordination (null when disabled).
  std::shared_ptr<MergeSkipState> skipState_;
  /// This bucket's assigned skip budget, applied to every row this merge
  /// emits (valid only after waitForSkipBudget returned true).
  int64_t mergeSkipRows_{0};
  /// Partition id this merge belongs to; used to publish the actual skipped
  /// row count to the shared skip state.
  int32_t skipPartitionId_{0};

  /// Publishes the number of rows this merge actually discarded to the shared
  /// skip state. No-op when offset skip is disabled. Must be called before the
  /// bucket end becomes visible to the ordered concat.
  void publishSkippedRows(int64_t rows);

 private:
  // Tracks the internal execution stats for a merge operator.
  struct Stats {
    // The time point that a merge operator starts reading from the streaming
    // source.
    uint64_t streamingSourceReadStartTimeUs{0};
    // The time point that a merge operator finishes read from the streaming
    // source. This includes the time that writes to the spilled source for
    // recursive merge when spilling is enabled for local merge.
    uint64_t streamingSourceReadEndTimeUs{0};
    // The time point that a merge operator finishes read from the spilled
    // source. This only applies when spilling is enabled for local merge.
    uint64_t spilledSourceReadEndTimeUs{0};
  };
  void recordMergeStats();

  // Starts the sources of the next partial merge run so upstream sinks can
  // make progress, without creating the source merger. The merger creation is
  // deferred to 'createSourceMerger' because the offset-skip budget must be
  // known first; the budget is assigned by the ordered concat only after the
  // preceding buckets drained, and waiting for the budget before starting the
  // sources would deadlock the upstream range-partitioned sink.
  void startNextMergeSourceGroup();

  // Creates the source merger for the group started by
  // 'startNextMergeSourceGroup', using the now-known skip budget.
  void createSourceMerger();

  // Returns true if needs to spill the merged source output if all sources can
  // not be merged at once.
  bool needSpill() const {
    return maxNumMergeSources_ < sources_.size();
  }

  void maybeSetupOutputSpiller();

  // Spill the output of a partial merge sources.
  void spill();

  // Invoked at the end for each partial merge run to ensure the order within
  // each spill file.
  void finishMergeSourceGroup();

  // Create spillMerger_ exactly once if spill has happened.
  void setupSpillMerger();

  RowVectorPtr getOutputFromSpill();

  RowVectorPtr getOutputFromSource();

  // Maximum number of rows in the output batch.
  const vector_size_t maxOutputBatchRows_;
  // Maximum number of bytes in the output batch.
  const uint64_t maxOutputBatchBytes_;
  const std::vector<SpillSortKey> sortingKeys_;

  Stats mergeStats_;

  RowVectorPtr output_;
  /// Number of rows accumulated in 'output_' so far.
  vector_size_t outputSize_{0};
  bool finished_{false};

  /// A list of blocking futures for sources. These are populates when a given
  /// source is blocked waiting for the next batch of data.
  std::vector<ContinueFuture> sourceBlockingFutures_;

  std::unique_ptr<SourceMerger> sourceMerger_;
  // Sources of the next merge group started by 'startNextMergeSourceGroup',
  // awaiting 'createSourceMerger'.
  std::vector<MergeSource*> pendingGroupSources_;
  std::shared_ptr<SpillMerger> spillMerger_;
  std::unique_ptr<MergeSpiller> mergeOutputSpiller_;
  // Number of total spilled rows, it must be equal to the input rows.
  uint64_t numSpilledRows_{0};
  // SpillFiles group for all the partial merge runs.
  std::vector<SpillFiles> spillFileGroups_;
};

/// A utility class for sort-merging data from upstream sources of the
/// `LocalMerge` operator. The `LocalMerge` operator may start only a portion of
/// the sources at a time to cap the memory usage, hence it might perform
/// multiple sort-merge operations with a subset of merge sources.
class SourceMerger {
 public:
  SourceMerger(
      const RowTypePtr& type,
      std::vector<std::unique_ptr<SourceStream>> sourceStreams,
      vector_size_t maxOutputBatchRows,
      uint64_t maxOutputBatchBytes,
      velox::memory::MemoryPool* pool,
      int64_t skipRows = 0);

  void isBlocked(std::vector<ContinueFuture>& sourceBlockingFutures) const;

  RowVectorPtr getOutput(
      std::vector<ContinueFuture>& sourceBlockingFutures,
      bool& atEnd);

  /// Number of leading rows this merger actually discarded. Valid while the
  /// merger is live; used by the owning merge operator to report the exact
  /// skipped count to the offset-skip coordination state.
  int64_t rowsSkipped() const {
    return rowsSkipped_;
  }

 private:
  void setOutputBatchSize();

  /// Creates the output vector. If a template is available from input data,
  /// creates output children with matching encodings to support FlatMapVector.
  RowVectorPtr createOutputVector();

  const RowTypePtr type_;
  const vector_size_t maxOutputBatchRows_;
  const uint64_t maxOutputBatchBytes_;
  const std::vector<SourceStream*> streams_;
  const std::unique_ptr<TreeOfLosers<SourceStream>> merger_;
  velox::memory::MemoryPool* const pool_;
  // Leading rows to discard without materializing output.
  int64_t skipRows_;
  // Rows actually discarded by this merger.
  int64_t rowsSkipped_{0};

  // The max number of rows in an output vector which is determined by
  // 'setOutputBatchSize'. The calculation is based on the actual estimated row
  // size and capped by 'maxOutputBatchRows_' and 'maxOutputBatchBytes_'.
  vector_size_t outputBatchRows_{0};

  // Reusable output vector.
  RowVectorPtr output_;
  // The number of rows in 'output_' vector.
  uint64_t outputRows_{0};
};

class SourceStream final : public MergeStream {
 public:
  SourceStream(
      MergeSource* source,
      const std::vector<SpillSortKey>& sortingKeys,
      uint32_t outputBatchSize)
      : source_{source},
        sortingKeys_{sortingKeys},
        outputRows_(outputBatchSize, false),
        sourceRows_(outputBatchSize) {
    keyColumns_.reserve(sortingKeys.size());
  }

  /// Returns true and appends a future to 'futures' if needs to wait for the
  /// source to produce data.
  bool isBlocked(std::vector<ContinueFuture>& futures) {
    if (needData_) {
      return fetchMoreData(futures);
    }
    return false;
  }

  bool hasData() const override {
    return !atEnd_;
  }

  /// Returns the current data batch from the source. Used for encoding
  /// detection to create output vectors with matching encodings.
  const RowVector* data() const {
    return data_.get();
  }

  // Returns the estimated row size based on the vector received from the
  // merge source.
  std::optional<int64_t> estimateRowSize() const {
    if (data_ == nullptr || data_->size() == 0) {
      return std::nullopt;
    }
    return data_->estimateFlatSize() / data_->size();
  }

  /// Returns true if current source row is less then current source row in
  /// 'other'.
  bool operator<(const MergeStream& other) const override;

  /// Advances to the next row. Returns true and appends a future to 'futures'
  /// if runs out of rows in the current batch and needs to wait for the
  /// source to produce the next batch. The return flag has the meaning of
  /// 'is-blocked'.
  bool pop(std::vector<ContinueFuture>& futures);

  /// Records the output row number for the current row. Returns true if
  /// current row is the last row in the current batch, in which case the
  /// caller must call 'copyToOutput' before calling pop(). The caller must
  /// call 'setOutputRow' before calling 'pop'. The output rows must
  /// monotonically increase in between calls to 'copyToOutput'.
  bool setOutputRow(vector_size_t row) {
    outputRows_.setValid(row, true);
    return currentSourceRow_ == data_->size() - 1;
  }

  /// Discards the current row without copying it out. Advances the copy
  /// position so a later 'copyToOutput' does not re-copy the discarded rows.
  /// Returns true and appends a future to 'futures' if runs out of rows in
  /// the current batch and needs to wait for the source to produce the next
  /// batch, mirroring 'pop'.
  bool skipCurrentRow(std::vector<ContinueFuture>& futures) {
    ++currentSourceRow_;
    ++firstSourceRow_;
    if (currentSourceRow_ == data_->size()) {
      VELOX_CHECK(!outputRows_.hasSelections());
      // The whole batch was discarded; the next batch starts from row zero.
      firstSourceRow_ = 0;
      return fetchMoreData(futures);
    }
    return false;
  }

  /// Called if either current row is the last row in the current batch or the
  /// caller accumulated enough output rows across all sources to produce an
  /// output batch.
  void copyToOutput(RowVectorPtr& output);

 private:
  bool fetchMoreData(std::vector<ContinueFuture>& futures);

  MergeSource* source_;

  const std::vector<SpillSortKey>& sortingKeys_;

  /// Ordered source rows.
  RowVectorPtr data_;

  /// Raw pointers to vectors corresponding to sorting key columns in the same
  /// order as 'sortingKeys_'.
  std::vector<BaseVector*> keyColumns_;

  /// Index of the current row.
  vector_size_t currentSourceRow_{0};

  /// True if source has been exhausted.
  bool atEnd_{false};

  /// True if ran out of rows in 'data_' and needs to wait for the future
  /// returned by 'source_->next()'.
  bool needData_{true};

  /// First source row that hasn't been copied out yet.
  vector_size_t firstSourceRow_{0};

  /// Output row numbers for source rows that haven't been copied out yet.
  SelectivityVector outputRows_;

  /// Reusable memory.
  std::vector<vector_size_t> sourceRows_;
};

/// A utility class for sort-merging data from data spilled by the `LocalMerge`
/// operator.
class SpillMerger : public std::enable_shared_from_this<SpillMerger> {
 public:
  SpillMerger(
      const std::vector<SpillSortKey>& sortingKeys,
      const RowTypePtr& type,
      std::vector<std::vector<std::unique_ptr<SpillReadFile>>>
          spillReadFilesGroup,
      vector_size_t maxOutputBatchRows,
      uint64_t maxOutputBatchBytes,
      int mergeSourceQueueSize,
      const common::SpillConfig* spillConfig,
      const std::shared_ptr<exec::SpillStats>& spillStats,
      velox::memory::MemoryPool* pool,
      int64_t skipRows = 0);

  ~SpillMerger();

  void start();

  RowVectorPtr getOutput(
      std::vector<ContinueFuture>& sourceBlockingFutures,
      bool& atEnd);

 private:
  static std::vector<std::shared_ptr<MergeSource>> createMergeSources(
      size_t numSpillSources,
      int queueSize);

  static std::vector<std::unique_ptr<BatchStream>> createBatchStreams(
      std::vector<std::vector<std::unique_ptr<SpillReadFile>>>
          spillReadFilesGroup);

  static std::unique_ptr<SourceMerger> createSourceMerger(
      const std::vector<SpillSortKey>& sortingKeys,
      const RowTypePtr& type,
      const std::vector<std::shared_ptr<MergeSource>>& sources,
      vector_size_t maxOutputBatchRows,
      uint64_t maxOutputBatchBytes,
      velox::memory::MemoryPool* pool,
      int64_t skipRows);

  void finishSource(size_t streamIdx) const;

  void readFromSpillFileStream(
      const std::weak_ptr<SpillMerger>& mergeHolder,
      size_t streamIdx);

  void scheduleAsyncSpillFileStreamReads();

  // Sets 'exception_' when an async reader throws.
  void setError(const std::exception_ptr& exception);

  // Returns true if any async reader has thrown an exception.
  bool hasError() const;

  // If any async reader has thrown an exception, rethrows it.
  void checkError();

  folly::Executor* const executor_;
  const std::shared_ptr<exec::SpillStats> spillStats_;
  const std::shared_ptr<memory::MemoryPool> pool_;
  const int64_t skipRows_;

  std::vector<std::shared_ptr<MergeSource>> sources_;
  std::vector<std::unique_ptr<BatchStream>> batchStreams_;
  std::unique_ptr<SourceMerger> sourceMerger_;
  mutable std::timed_mutex mutex_;
  std::exception_ptr exception_ = nullptr;
};

// LocalMerge merges its source's output into a single stream of
// sorted rows. It runs single threaded. The sources may run multi-threaded and
// in the same task.
class LocalMerge : public Merge {
 public:
  LocalMerge(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::LocalMergeNode>& localMergeNode);

 protected:
  BlockingReason addMergeSources(ContinueFuture* future) override;

  bool waitForSkipBudget(ContinueFuture* future) override;

 private:
  // For a range-partitioned merge, this driver merges only the sources of its
  // own partition (one bucket per driver).
  const int32_t partitionId_{0};
  const bool rangePartitioned_{false};
};

/// Routes the rows of a sorted run into range-bucket merge sources: the sink
/// of the source pipeline of a range-partitioned LocalMergeNode. Each row is
/// assigned to the bucket of its first sort key using the partition
/// boundaries; within a bucket the rows of every sorted run stay sorted, so
/// each bucket's merge driver re-merges a set of sorted slices.
class RangePartitionedMergeSink : public Operator {
 public:
  RangePartitionedMergeSink(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::LocalMergeNode>& localMergeNode);

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override {
    return nullptr;
  }

  bool needsInput() const override {
    return true;
  }

  void noMoreInput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override {
    return finished_;
  }

  bool canAddDynamicFilter() const override {
    return false;
  }

 private:
  void enqueueRowVector(int32_t partition, RowVectorPtr vector);

  const int32_t numPartitions_;
  // Ascending partition boundaries; rows with key < boundaries[j] go to
  // bucket j (nulls go to the nullsFirst bucket).
  const std::vector<facebook::velox::variant> boundaries_;
  const bool nullsFirst_;
  const column_index_t keyChannel_;
  std::vector<std::shared_ptr<MergeSource>> sources_;
  std::vector<ContinueFuture> blockingFutures_;
  // Buckets this sink has already signaled end-of-data to. The sink's routing
  // is monotone (a sorted run's rows advance through the ascending range
  // buckets), so once the current batch's lowest bucket moves past a bucket,
  // no more rows will ever be routed to it: signal atEnd early so the bucket's
  // merge does not wait for this sink to fully finish.
  std::vector<bool> partitionEnded_;
  bool started_{false};
  bool finished_{false};
};

/// Concatenates its merge sources in partition order. Used as the final stage
/// of a range-partitioned merge: the bucket outputs are disjoint ordered
/// ranges, so draining them in order yields the globally sorted stream without
/// any comparisons.
class OrderedConcat : public SourceOperator {
 public:
  OrderedConcat(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::OrderedConcatNode>& concatNode);

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  RowVectorPtr getOutput() override;

  void close() override;

 private:
  bool addMergeSources(ContinueFuture* future);

  void startNextSource(ContinueFuture* future);

  // Merge sources sorted by partition id.
  std::vector<std::shared_ptr<MergeSource>> sources_;
  size_t currentSource_{0};
  bool allSourcesAdded_{false};
  std::optional<ContinueFuture> blockedFuture_;
  bool finished_{false};
  // Offset-skip coordination for the range-partitioned merge below (null
  // when disabled).
  std::shared_ptr<MergeSkipState> skipState_;
};

// MergeExchange merges its sources' outputs into a single stream of
// sorted rows similar to local merge. However, the sources are splits
// and may be generated by a different task.
class MergeExchange : public Merge {
 public:
  MergeExchange(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::MergeExchangeNode>& orderByNode);

  VectorSerde* serde() const {
    return serde_;
  }

  VectorSerde::Options* serdeOptions() const {
    return serdeOptions_.get();
  }

  void close() override;

 protected:
  BlockingReason addMergeSources(ContinueFuture* future) override;

 private:
  VectorSerde* const serde_;
  const std::unique_ptr<VectorSerde::Options> serdeOptions_;
  bool noMoreSplits_ = false;
  // Task Ids from all the splits we took to process so far.
  std::vector<std::string> remoteSourceTaskIds_;
};

} // namespace facebook::velox::exec
