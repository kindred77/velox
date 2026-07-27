#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>

namespace facebook::velox::perf {

struct Counter {
    std::atomic<uint64_t> total{0};
    std::atomic<uint64_t> cnt{0};

    void add(uint64_t us) {
        total.fetch_add(us, std::memory_order_relaxed);
        cnt.fetch_add(1, std::memory_order_relaxed);
    }
};

struct ScopedTimer {
    using clock = std::chrono::steady_clock;
    clock::time_point start_;
    Counter* counter_;

    ScopedTimer(Counter* c) : start_(clock::now()), counter_(c) {}
    ~ScopedTimer() {
        counter_->add(std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start_).count());
    }
};

struct PerfTracer {
    // TableScan
    Counter ts_getOutputTotal;
    Counter ts_getSplit;
    Counter ts_dataSourceNext;
    Counter ts_postNext;
    Counter ts_createDataSource;
    Counter ts_addSplit;
    Counter ts_preloadMove;
    Counter ts_setFromDataSource;

    // FileSplitReader::createReader breakdown
    Counter cr_fhGenerate;
    Counter cr_createBufferedInput;
    Counter cr_createParquetReader;

    // FileSplitReader::prepareSplit breakdown
    Counter ps_createReader;
    Counter ps_checkEmpty;
    Counter ps_createRowReader;

    // ReaderBase (parquet footer/schema) breakdown
    Counter rb_loadFileMetaData;
    Counter rb_initializeSchema;

    // ParquetRowReader::Impl constructor breakdown
    Counter prr_buildColumnReader;
    Counter prr_filterRowGroups;
    Counter prr_advanceToNextRG;


    // GroupingSet (global aggregation)
    Counter gs_addGlobalAggTotal;
    Counter gs_addRawInput;

    // Driver (operator-level)
    Counter op_getOutput;
    Counter op_addInput;

    static PerfTracer& instance() {
        static PerfTracer inst;
        return inst;
    }

    void dumpAll() {
        auto d = [](const char* name, Counter& c) {
            uint64_t cnt = c.cnt.load();
            if (!cnt) return;
            std::cerr << "[PERF] " << name
                      << " total=" << c.total.load() << " us"
                      << " cnt=" << cnt
                      << " avg=" << (c.total.load() / cnt) << " us"
                      << std::endl;
        };
        std::cerr << "======= PERF COUNTERS =======" << std::endl;
        d("ts_getOutputTotal", ts_getOutputTotal);
        d("ts_getSplit", ts_getSplit);
        d("ts_dataSourceNext", ts_dataSourceNext);
        d("ts_postNext", ts_postNext);
        d("ts_createDataSource", ts_createDataSource);
        d("ts_addSplit", ts_addSplit);
        d("ts_preloadMove", ts_preloadMove);
        d("ts_setFromDataSource", ts_setFromDataSource);
        d("ps_createReader", ps_createReader);
        d("cr_fhGenerate", cr_fhGenerate);
        d("cr_createBufferedInput", cr_createBufferedInput);
        d("cr_createParquetReader", cr_createParquetReader);
        d("ps_checkEmpty", ps_checkEmpty);
        d("ps_createRowReader", ps_createRowReader);
        d("rb_loadFileMetaData", rb_loadFileMetaData);
        d("rb_initializeSchema", rb_initializeSchema);
        d("prr_buildColumnReader", prr_buildColumnReader);
        d("prr_filterRowGroups", prr_filterRowGroups);
        d("prr_advanceToNextRG", prr_advanceToNextRG);
        d("gs_addGlobalAggTotal", gs_addGlobalAggTotal);
        d("gs_addRawInput", gs_addRawInput);
        d("op_getOutput", op_getOutput);
        d("op_addInput", op_addInput);
    }
};

} // namespace facebook::velox::perf
