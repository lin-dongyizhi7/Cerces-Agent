#ifndef METRICS_COMMON_H
#define METRICS_COMMON_H

#include <string>
#include <chrono>
#include <vector>
#include <map>
#include <cmath>

// 指标数据结构
struct Metrics {
    // 时间戳
    std::chrono::system_clock::time_point timestamp;
    
    // 全过程监控指标 (T1-T15)
    double power;                    // T1: 功率 (W)
    double temperature;               // T2: 温度 (℃)
    double ai_core_usage;             // T3: AI Core 占用率 (%)
    double ai_cpu_usage;              // T4: AI Cpu 占用率 (%)
    double ctrl_cpu_usage;            // T5: Ctrl Cpu 占用率 (%)
    double memory_usage;              // T6: 内存占用率 (%)
    double memory_bandwidth_usage;    // T7: 内存带宽占用率 (%)
    double python_gc_time;            // T8: Python的GC耗时 (ms)
    
    // T9-T14: 内存相关aclrt函数吞吐量 (MB/s)
    double aclrtMemcpyAsync_throughput;      // T9
    double aclrtMemcpy2dAsync_throughput;    // T10
    double aclrtFree_throughput;             // T11
    double aclrtFreeHost_throughput;         // T12
    double aclrtMalloc_throughput;           // T13
    double aclrtMallocAsync_throughput;      // T14
    
    double aclrtLaunchKernel_latency;        // T15: aclrtLaunchKernel启动时延 (μs)
    
    // 数据加载阶段 (D1)
    double dataloader_throughput;            // D1: DataLoader吞吐量 (batch/s)
    
    // 前向传播阶段 (F1-F4)
    double aclnnFlashAttentionScore_time;   // F1: 执行时间 (s)
    double aclnnFlashAttentionScore_tflops; // F1: 算力 (TFLOPs)
    double aclnnMatmul_tflops;               // F2: 算力 (TFLOPs)
    double aclnnBatchMatMul_tflops;         // F3: 算力 (TFLOPs)
    double aclnnFFN_time;                    // F4: 执行时间 (s)
    double aclnnFFN_tflops;                  // F4: 算力 (TFLOPs)
    
    // 反向计算阶段 (B1-B5)
    double aclnnFlashAttentionScoreGrad_tflops; // B1: 算力 (TFLOPs)
    double aclnnMatmul_grad_tflops;              // B2: 算力 (TFLOPs)
    double aclnnBatchMatMul_grad_tflops;         // B3: 算力 (TFLOPs)
    double torch_autograd_backward_time;         // B4: 耗时 (ms)
    double torch_autograd_grad_time;             // B5: 耗时 (ms)
    
    // 梯度同步&参数更新阶段 (G1-G7)
    double hcclAllReduce_bandwidth;          // G1: 通信效率 (GB/s)
    double hcclBroadcast_bandwidth;          // G2: 通信效率 (GB/s)
    double hcclAllGather_bandwidth;          // G3: 通信效率 (GB/s)
    double hcclReduceScatter_bandwidth;      // G4: 通信效率 (GB/s)
    double aclrtSynchronizeStream_time;       // G5: 流同步耗时 (μs)
    double aclrtSynchronizeEvent_time;        // G6: 事件同步耗时 (μs)
    double aclrtStreamWaitEvent_time;        // G7: 流等待事件耗时 (μs)
    
    // 构造函数，初始化所有指标为0
    Metrics() {
        timestamp = std::chrono::system_clock::now();
        power = 0.0;
        temperature = 0.0;
        ai_core_usage = 0.0;
        ai_cpu_usage = 0.0;
        ctrl_cpu_usage = 0.0;
        memory_usage = 0.0;
        memory_bandwidth_usage = 0.0;
        python_gc_time = 0.0;
        aclrtMemcpyAsync_throughput = 0.0;
        aclrtMemcpy2dAsync_throughput = 0.0;
        aclrtFree_throughput = 0.0;
        aclrtFreeHost_throughput = 0.0;
        aclrtMalloc_throughput = 0.0;
        aclrtMallocAsync_throughput = 0.0;
        aclrtLaunchKernel_latency = 0.0;
        dataloader_throughput = 0.0;
        aclnnFlashAttentionScore_time = 0.0;
        aclnnFlashAttentionScore_tflops = 0.0;
        aclnnMatmul_tflops = 0.0;
        aclnnBatchMatMul_tflops = 0.0;
        aclnnFFN_time = 0.0;
        aclnnFFN_tflops = 0.0;
        aclnnFlashAttentionScoreGrad_tflops = 0.0;
        aclnnMatmul_grad_tflops = 0.0;
        aclnnBatchMatMul_grad_tflops = 0.0;
        torch_autograd_backward_time = 0.0;
        torch_autograd_grad_time = 0.0;
        hcclAllReduce_bandwidth = 0.0;
        hcclBroadcast_bandwidth = 0.0;
        hcclAllGather_bandwidth = 0.0;
        hcclReduceScatter_bandwidth = 0.0;
        aclrtSynchronizeStream_time = 0.0;
        aclrtSynchronizeEvent_time = 0.0;
        aclrtStreamWaitEvent_time = 0.0;
    }
};

// 异常检测结果
struct AnomalyResult {
    bool has_anomaly;
    std::string rule_name;
    std::string metric_name;
    std::string description;
    double current_value;
    double threshold_value;
    
    AnomalyResult() : has_anomaly(false) {}
};

// 动态基线数据结构
struct Baseline {
    double mean;
    double std_dev;
    std::vector<double> history;
    size_t window_size;
    
    Baseline(size_t window = 100) : mean(0.0), std_dev(0.0), window_size(window) {}
    
    void update(double value) {
        history.push_back(value);
        if (history.size() > window_size) {
            history.erase(history.begin());
        }
        calculate_stats();
    }
    
    void calculate_stats() {
        if (history.empty()) {
            mean = 0.0;
            std_dev = 0.0;
            return;
        }
        
        double sum = 0.0;
        for (double v : history) {
            sum += v;
        }
        mean = sum / history.size();
        
        double variance = 0.0;
        for (double v : history) {
            variance += (v - mean) * (v - mean);
        }
        std_dev = std::sqrt(variance / history.size());
    }
};

// CUSUM统计量
struct CUSUMStats {
    double cumulative_sum;
    double reference_offset;
    double decision_threshold;
    
    CUSUMStats(double offset = 0.5, double threshold = 5.0) 
        : cumulative_sum(0.0), reference_offset(offset), decision_threshold(threshold) {}
    
    void update(double value, double baseline_mean, double baseline_std) {
        double normalized = (value - baseline_mean) / baseline_std;
        cumulative_sum = std::max(0.0, cumulative_sum + normalized - reference_offset);
    }
    
    bool is_anomaly() const {
        return cumulative_sum > decision_threshold;
    }
    
    void reset() {
        cumulative_sum = 0.0;
    }
};

#endif // METRICS_COMMON_H

