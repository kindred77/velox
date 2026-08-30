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

#include "velox/dwio/parquet/reader/ParquetData.h"

#include <charconv>
#include <cstdlib>

#include "velox/common/time/Timer.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/parquet/reader/ParquetStatsContext.h"

namespace facebook::velox::parquet {

namespace {

// E0-B temporary diagnostic gate: maximum chunk size (bytes) eligible for the
// direct single-use read path; 0 disables it and falls back to the cached
// stream path for every chunk. Remove together with the A/B closure.
uint64_t directChunkMaxBytes() {
  const char* value = std::getenv("MYGORCA_PARQUET_DIRECT_CHUNK_MAX");
  constexpr uint64_t kDefaultMax = 8ULL << 20; // 8MB = default load quantum.
  if (value == nullptr) {
    return kDefaultMax;
  }
  uint64_t bytes = 0;
  const char* end = value + std::char_traits<char>::length(value);
  const auto [ptr, error] = std::from_chars(value, end, bytes);
  if (error != std::errc{} || ptr != end) {
    return kDefaultMax;
  }
  return bytes;
}

} // namespace

std::unique_ptr<dwio::common::FormatData> ParquetParams::toFormatData(
    const std::shared_ptr<const dwio::common::TypeWithId>& type,
    const common::ScanSpec& /*scanSpec*/) {
  return std::make_unique<ParquetData>(
      type, metaData_, pool(), runtimeStatistics(), sessionTimezone_);
}

void ParquetData::filterRowGroups(
    const common::ScanSpec& scanSpec,
    uint64_t /*rowsPerRowGroup*/,
    const dwio::common::StatsContext& writerContext,
    FilterRowGroupsResult& result) {
  auto parquetStatsContext =
      reinterpret_cast<const ParquetStatsContext*>(&writerContext);
  if (type_->parquetType_.has_value() &&
      parquetStatsContext->shouldIgnoreStatistics(
          type_->parquetType_.value())) {
    return;
  }
  result.totalCount =
      std::max<int>(result.totalCount, fileMetaDataPtr_.numRowGroups());
  auto nwords = bits::nwords(result.totalCount);
  if (result.filterResult.size() < nwords) {
    result.filterResult.resize(nwords);
  }
  auto metadataFiltersStartIndex = result.metadataFilterResults.size();
  for (int i = 0; i < scanSpec.numMetadataFilters(); ++i) {
    result.metadataFilterResults.emplace_back(
        scanSpec.metadataFilterNodeAt(i), std::vector<uint64_t>(nwords));
  }
  if (scanSpec.filter() || scanSpec.numMetadataFilters() > 0) {
    for (auto i = 0; i < fileMetaDataPtr_.numRowGroups(); ++i) {
      // Already excluded by another column or by the caller (e.g. row group
      // outside the split range, empty row group). Skip statistics build and
      // testFilter. The MetadataFilter::eval call ORs into filterResult, so
      // leaving the per-leaf metadata bits at 0 here is harmless: filterResult
      // already has the bit set.
      if (bits::isBitSet(result.filterResult.data(), i)) {
        continue;
      }
      if (scanSpec.filter() && !rowGroupMatches(i, scanSpec.filter())) {
        bits::setBit(result.filterResult.data(), i);
        continue;
      }
      for (int j = 0; j < scanSpec.numMetadataFilters(); ++j) {
        auto* metadataFilter = scanSpec.metadataFilterAt(j);
        if (!rowGroupMatches(i, metadataFilter)) {
          bits::setBit(
              result.metadataFilterResults[metadataFiltersStartIndex + j]
                  .second.data(),
              i);
        }
      }
    }
  }
}

bool ParquetData::rowGroupMatches(
    uint32_t rowGroupId,
    const common::Filter* filter) {
  auto column = type_->column();
  auto type = type_->type();
  auto rowGroup = fileMetaDataPtr_.rowGroup(rowGroupId);
  assert(rowGroup.numColumns() != 0);

  if (!filter) {
    return true;
  }

  auto columnChunk = rowGroup.columnChunk(column);
  if (columnChunk.hasStatistics()) {
    auto columnStats =
        columnChunk.getColumnStatistics(type, rowGroup.numRows());
    return testFilter(filter, columnStats.get(), rowGroup.numRows(), type);
  }
  return true;
}

void ParquetData::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input) {
  auto chunk = fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
  streams_.resize(fileMetaDataPtr_.numRowGroups());
  directChunks_.resize(fileMetaDataPtr_.numRowGroups());
  VELOX_CHECK(
      chunk.hasMetadata(),
      "ColumnMetaData does not exist for schema Id ",
      type_->column());
  ;

  uint64_t chunkReadOffset = chunk.dataPageOffset();
  if (chunk.hasDictionaryPageOffset() && chunk.dictionaryPageOffset() >= 4) {
    // this assumes the data pages follow the dict pages directly.
    chunkReadOffset = chunk.dictionaryPageOffset();
  }

  uint64_t readSize =
      (chunk.compression() == common::CompressionKind::CompressionKind_NONE)
      ? chunk.totalUncompressedSize()
      : chunk.totalCompressedSize();

  if (readSize > 0 && readSize <= directChunkMaxBytes()) {
    // E0-B: skip the cache enqueue; the chunk is read directly into a reused
    // buffer on seekToRowGroup(). Keeps the chunk metadata in the footer (T3
    // byte cache) and only bypasses AsyncDataCache for single-use page data.
    directInput_ = input.getInputStream();
    directChunks_[index] = std::make_pair(chunkReadOffset, readSize);
    return;
  }

  auto id = dwio::common::StreamIdentifier(type_->column());
  streams_[index] = input.enqueue({chunkReadOffset, readSize}, &id);
}

dwio::common::PositionProvider ParquetData::seekToRowGroup(int64_t index) {
  static std::vector<uint64_t> empty;
  VELOX_CHECK_LT(index, streams_.size());
  if (directChunks_[index].has_value()) {
    // E0-B: read the whole chunk with one pread into the reused buffer and
    // serve the pages from memory. The old reader is destroyed first so its
    // shared buffers are returned to bufferCache_ before reuse.
    reader_.reset();
    const auto [chunkOffset, chunkSize] = *directChunks_[index];
    dwio::common::ensureCapacity<char>(directChunkBuffer_, chunkSize, &pool_);
    directChunkBuffer_->setSize(chunkSize);
    std::vector<folly::Range<char*>> ranges = {folly::Range<char*>(
        directChunkBuffer_->asMutable<char>(), chunkSize)};
    uint64_t readUs{0};
    {
      MicrosecondWallTimer timer(&readUs);
      directInput_->read(ranges, chunkOffset, dwio::common::LogType::FILE);
    }
    // Keep the DWIO-level IO metrics in sync (ReadFileInputStream only
    // updates the ReadFile-layer IoStats): storageReadBytes, read latency and
    // raw bytes touched, mirroring DirectInputStream::loadSync.
    if (auto* ioStats = directInput_->getStats()) {
      ioStats->read().increment(chunkSize);
      ioStats->incRawBytesRead(chunkSize);
      ioStats->queryThreadIoLatencyUs().increment(readUs);
      ioStats->storageReadLatencyUs().increment(readUs);
      ioStats->incTotalScanTimeNs(readUs * 1'000);
    }
    stats_.pageLoadTimeNs.increment(readUs * 1'000);
    auto metadata =
        fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
    auto stream = std::make_unique<dwio::common::SeekableArrayInputStream>(
        directChunkBuffer_->as<char>(), chunkSize);
    reader_ = std::make_unique<PageReader>(
        std::move(stream),
        pool_,
        type_,
        metadata.compression(),
        metadata.totalCompressedSize(),
        stats_,
        sessionTimezone_,
        &bufferCache_);
    return dwio::common::PositionProvider(empty);
  }
  VELOX_CHECK(streams_[index], "Stream not enqueued for column");
  auto metadata = fileMetaDataPtr_.rowGroup(index).columnChunk(type_->column());
  reader_ = std::make_unique<PageReader>(
      std::move(streams_[index]),
      pool_,
      type_,
      metadata.compression(),
      metadata.totalCompressedSize(),
      stats_,
      sessionTimezone_,
      &bufferCache_);
  return dwio::common::PositionProvider(empty);
}

std::pair<int64_t, int64_t> ParquetData::getRowGroupRegion(
    uint32_t index) const {
  auto rowGroup = fileMetaDataPtr_.rowGroup(index);

  VELOX_CHECK_GT(rowGroup.numColumns(), 0);
  auto fileOffset = (rowGroup.hasFileOffset() && rowGroup.fileOffset() != 0)
      ? rowGroup.fileOffset()
      : rowGroup.columnChunk(0).hasDictionaryPageOffset()
      ? rowGroup.columnChunk(0).dictionaryPageOffset()
      : rowGroup.columnChunk(0).dataPageOffset();
  VELOX_CHECK_GT(fileOffset, 0);

  auto length = rowGroup.hasTotalCompressedSize()
      ? rowGroup.totalCompressedSize()
      : rowGroup.totalByteSize();

  return {fileOffset, length};
}

} // namespace facebook::velox::parquet
