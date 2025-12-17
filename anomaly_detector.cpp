#include "metrics_common.h"
#include <iostream>
#include <queue>
#include <map>
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>
#include <ctime>
#include <cstring>
#include <thread>
#include <random>

class AnomalyDetector {
private:
    struct ThresholdConfig {
        double upper_limit;
        double lower_limit;
        int consecutive_periods;
        
        ThresholdConfig(double upper = std::numeric_limits<double>::max(),
                       double lower = std::numeric_limits<double>::min(),
                       int periods = 3)
            : upper_limit(upper), lower_limit(lower), consecutive_periods(periods) {}
    };
    
    // 指标阈值映射
    std::map<std::string, ThresholdConfig> thresholds;
    
    // 连续越界计数器
    std::map<std::string, int> violation_counters;
    
    // 动态基线（滑动窗口）
    std::map<std::string, Baseline> baselines;
    
    // CUSUM统计量
    std::map<std::string, CUSUMStats> cusum_stats;
    
    // 训练吞吐量相关（R3）
    std::queue<double> throughput_history;
    size_t throughput_window_size;
    int warmup_steps;
    int current_step;
    bool is_warmup_phase;
    
    // 计算密集型核函数基准FLOPS（R4）
    std::map<std::string, double> baseline_flops;
    std::map<std::string, std::vector<double>> flops_history;
    size_t flops_window_size;
    
    // 通信带宽相关（R5-R6）
    std::map<std::string, std::vector<double>> comm_bandwidth_by_group;
    std::map<std::string, std::vector<double>> comm_bandwidth_by_rank;
    
    // 次要NPU操作相关（R7）
    double total_step_time;
    double monitored_kernel_time;
    
    // 核启动延迟相关（R8）
    std::vector<double> kernel_launch_latencies;
    
    // 内存拷贝速率相关（R9）
    double h2d_bandwidth;
    double d2h_bandwidth;
    
    // 步间CPU操作相关（R10）
    double inter_step_cpu_time;
    double step_total_time;
    
    double sensitivity_k;
    
public:
    AnomalyDetector() 
        : throughput_window_size(50),
          warmup_steps(100),
          current_step(0),
          is_warmup_phase(true),
          flops_window_size(100),
          total_step_time(0.0),
          monitored_kernel_time(0.0),
          h2d_bandwidth(0.0),
          d2h_bandwidth(0.0),
          inter_step_cpu_time(0.0),
          step_total_time(0.0),
          sensitivity_k(2.5) {
        
        initializeThresholds();
        initializeBaselineFLOPS();
    }
    
    void initializeThresholds() {
        // T1: 功率 (W) - 上限300W，下限200W
        thresholds["power"] = ThresholdConfig(300.0, 200.0, 3);
        
        // T2: 温度 (℃) - 上限85℃，下限40℃
        thresholds["temperature"] = ThresholdConfig(85.0, 40.0, 3);
        
        // T3: AI Core占用率 (%) - 上限100%，下限50%
        thresholds["ai_core_usage"] = ThresholdConfig(100.0, 50.0, 3);
        
        // T4: AI Cpu占用率 (%) - 上限80%
        thresholds["ai_cpu_usage"] = ThresholdConfig(80.0, 0.0, 3);
        
        // T5: Ctrl Cpu占用率 (%) - 上限90%
        thresholds["ctrl_cpu_usage"] = ThresholdConfig(90.0, 0.0, 3);
        
        // T6: 内存占用率 (%) - 上限90%
        thresholds["memory_usage"] = ThresholdConfig(90.0, 0.0, 3);
        
        // T7: 内存带宽占用率 (%) - 上限95%
        thresholds["memory_bandwidth_usage"] = ThresholdConfig(95.0, 0.0, 3);
        
        // T8: Python GC耗时 (ms) - 上限500ms
        thresholds["python_gc_time"] = ThresholdConfig(500.0, 0.0, 3);
        
        baselines["power"] = Baseline(100);
        baselines["temperature"] = Baseline(100);
        baselines["ai_core_usage"] = Baseline(100);
        
        cusum_stats["power"] = CUSUMStats(0.5, 5.0);
        cusum_stats["temperature"] = CUSUMStats(0.5, 5.0);
        cusum_stats["ai_core_usage"] = CUSUMStats(0.5, 5.0);
    }
    
    void initializeBaselineFLOPS() {
        baseline_flops["aclnnFlashAttentionScore"] = 100.0;  // TFLOPs
        baseline_flops["aclnnMatmul"] = 150.0;
        baseline_flops["aclnnBatchMatMul"] = 120.0;
        baseline_flops["aclnnFFN"] = 80.0;
        baseline_flops["aclnnFlashAttentionScoreGrad"] = 90.0;
        baseline_flops["aclnnMatmul_grad"] = 140.0;
        baseline_flops["aclnnBatchMatMul_grad"] = 110.0;
    }
    
    // R1: 静态阈值越界检测
    AnomalyResult checkStaticThreshold(const std::string& metric_name, double value) {
        AnomalyResult result;
        
        if (thresholds.find(metric_name) == thresholds.end()) {
            return result;
        }
        
        const ThresholdConfig& config = thresholds[metric_name];
        
        if (value > config.upper_limit || value < config.lower_limit) {
            violation_counters[metric_name]++;
            
            if (violation_counters[metric_name] >= config.consecutive_periods) {
                result.has_anomaly = true;
                result.rule_name = "R1";
                result.metric_name = metric_name;
                result.current_value = value;
                result.threshold_value = (value > config.upper_limit) ? config.upper_limit : config.lower_limit;
                
                if (value > config.upper_limit) {
                    result.description = "超过上限阈值: " + std::to_string(config.upper_limit);
                } else {
                    result.description = "低于下限阈值: " + std::to_string(config.lower_limit);
                }
            }
        } else {
            violation_counters[metric_name] = 0;
        }
        
        return result;
    }
    
    // R2: CUSUM动态趋势偏移检测
    AnomalyResult checkCUSUM(const std::string& metric_name, double value) {
        AnomalyResult result;
        
        if (baselines.find(metric_name) == baselines.end() ||
            cusum_stats.find(metric_name) == cusum_stats.end()) {
            return result;
        }
        
        Baseline& baseline = baselines[metric_name];
        CUSUMStats& cusum = cusum_stats[metric_name];
        
        baseline.update(value);
        
        if (baseline.history.size() > 10) {
            cusum.update(value, baseline.mean, baseline.std_dev);
            
            if (cusum.is_anomaly()) {
                result.has_anomaly = true;
                result.rule_name = "R2";
                result.metric_name = metric_name;
                result.current_value = value;
                result.threshold_value = cusum.decision_threshold;
                result.description = "CUSUM累积统计量超过阈值，检测到持续性偏移趋势";
            }
        }
        
        return result;
    }
    
    // R3: 对比训练吞吐量数据
    AnomalyResult checkThroughput(double throughput) {
        AnomalyResult result;
        current_step++;
        
        if (current_step <= warmup_steps) {
            is_warmup_phase = true;
            return result;
        }
        
        is_warmup_phase = false;
        throughput_history.push(throughput);
        
        if (throughput_history.size() > throughput_window_size) {
            throughput_history.pop();
        }
        
        if (throughput_history.size() < 10) {
            return result;
        }
        
        std::vector<double> values;
        std::queue<double> temp_queue = throughput_history;
        while (!temp_queue.empty()) {
            values.push_back(temp_queue.front());
            temp_queue.pop();
        }
        
        double mean = 0.0;
        for (double v : values) {
            mean += v;
        }
        mean /= values.size();
        
        double variance = 0.0;
        for (double v : values) {
            variance += (v - mean) * (v - mean);
        }
        double std_dev = std::sqrt(variance / values.size());
        
        double upper_bound = mean + sensitivity_k * std_dev;
        double lower_bound = mean - sensitivity_k * std_dev;
        
        if (throughput > upper_bound || throughput < lower_bound) {
            result.has_anomaly = true;
            result.rule_name = "R3";
            result.metric_name = "DataLoader_throughput";
            result.current_value = throughput;
            result.threshold_value = (throughput > upper_bound) ? upper_bound : lower_bound;
            result.description = "训练吞吐量超出正常范围 [" + 
                               std::to_string(lower_bound) + ", " + 
                               std::to_string(upper_bound) + "]";
        }
        
        return result;
    }
    
    // R4: 对比基准FLOPS数据
    AnomalyResult checkFLOPS(const std::string& kernel_name, double actual_flops) {
        AnomalyResult result;
        
        if (baseline_flops.find(kernel_name) == baseline_flops.end()) {
            return result;
        }
        
        double baseline = baseline_flops[kernel_name];
        double relative_performance = actual_flops / baseline;
        
        flops_history[kernel_name].push_back(relative_performance);
        if (flops_history[kernel_name].size() > flops_window_size) {
            flops_history[kernel_name].erase(flops_history[kernel_name].begin());
        }
        
        if (flops_history[kernel_name].size() < 20) {
            return result;
        }
        
        std::vector<double> sorted = flops_history[kernel_name];
        std::sort(sorted.begin(), sorted.end());
        
        size_t n = sorted.size();
        double q25 = sorted[n * 0.25];
        double q75 = sorted[n * 0.75];
        double iqr = q75 - q25;
        
        double upper_bound = q75 + 1.5 * iqr;
        double lower_bound = q25 - 1.5 * iqr;
        
        if (relative_performance > upper_bound || relative_performance < lower_bound) {
            result.has_anomaly = true;
            result.rule_name = "R4";
            result.metric_name = kernel_name;
            result.current_value = actual_flops;
            result.threshold_value = baseline * ((relative_performance > upper_bound) ? upper_bound : lower_bound);
            result.description = "计算密集型核函数性能波动异常，相对性能: " + 
                               std::to_string(relative_performance * 100) + "%";
        }
        
        return result;
    }
    
    // R5: 组内单rank异常检验
    AnomalyResult checkIntraGroupAnomaly(const std::string& group_id, int rank_id, double bandwidth) {
        AnomalyResult result;
        
        comm_bandwidth_by_rank[group_id + "_" + std::to_string(rank_id)].push_back(bandwidth);
        
        std::vector<double> group_bandwidths;
        for (auto& pair : comm_bandwidth_by_rank) {
            if (pair.first.find(group_id) == 0) {
                if (!pair.second.empty()) {
                    group_bandwidths.push_back(pair.second.back());
                }
            }
        }
        
        if (group_bandwidths.size() < 2) {
            return result;
        }
        
        double mean = 0.0;
        for (double b : group_bandwidths) {
            mean += b;
        }
        mean /= group_bandwidths.size();
        
        double variance = 0.0;
        for (double b : group_bandwidths) {
            variance += (b - mean) * (b - mean);
        }
        double std_dev = std::sqrt(variance / group_bandwidths.size());
        
        // 判断是否偏离组均值±3σ
        if (std::abs(bandwidth - mean) > 3 * std_dev) {
            result.has_anomaly = true;
            result.rule_name = "R5";
            result.metric_name = "Comm_bandwidth_" + group_id + "_rank" + std::to_string(rank_id);
            result.current_value = bandwidth;
            result.threshold_value = mean;
            result.description = "组内单rank通信带宽异常，偏离组均值: " + 
                               std::to_string(std::abs(bandwidth - mean)) + " GB/s";
        }
        
        return result;
    }
    
    // R6: 跨DP组异常检验
    AnomalyResult checkInterGroupAnomaly(const std::string& group_id, double avg_bandwidth) {
        AnomalyResult result;
        
        comm_bandwidth_by_group[group_id].push_back(avg_bandwidth);
        
        // 计算所有正常组的平均带宽
        std::vector<double> normal_group_bandwidths;
        for (auto& pair : comm_bandwidth_by_group) {
            if (!pair.second.empty()) {
                normal_group_bandwidths.push_back(pair.second.back());
            }
        }
        
        if (normal_group_bandwidths.size() < 2) {
            return result;
        }
        
        double overall_mean = 0.0;
        for (double b : normal_group_bandwidths) {
            overall_mean += b;
        }
        overall_mean /= normal_group_bandwidths.size();
        
        double variance = 0.0;
        for (double b : normal_group_bandwidths) {
            variance += (b - overall_mean) * (b - overall_mean);
        }
        double std_dev = std::sqrt(variance / normal_group_bandwidths.size());
        
        if (std::abs(avg_bandwidth - overall_mean) > 3 * std_dev) {
            result.has_anomaly = true;
            result.rule_name = "R6";
            result.metric_name = "Comm_bandwidth_group_" + group_id;
            result.current_value = avg_bandwidth;
            result.threshold_value = overall_mean;
            result.description = "跨DP组通信异常，整体带宽偏离: " + 
                               std::to_string(std::abs(avg_bandwidth - overall_mean)) + " GB/s";
        }
        
        return result;
    }
    
    // R7: 次要NPU操作异常检验
    AnomalyResult checkMinorKernelAnomaly(double step_time, double monitored_time) {
        AnomalyResult result;
        
        double effective_time = step_time;  // 简化：假设有效计算时段等于step_time
        double minor_kernel_time = step_time - monitored_time;
        double v_minority = (minor_kernel_time / effective_time) * 100.0;
        
        // 阈值：次要内核空窗占比不应超过40%
        if (v_minority > 40.0) {
            result.has_anomaly = true;
            result.rule_name = "R7";
            result.metric_name = "Minor_kernel_ratio";
            result.current_value = v_minority;
            result.threshold_value = 40.0;
            result.description = "次要NPU内核空窗占比异常: " + std::to_string(v_minority) + "%";
        }
        
        return result;
    }
    
    // R8: 核启动延迟分布判断
    AnomalyResult checkKernelLaunchLatency(double latency) {
        AnomalyResult result;
        
        kernel_launch_latencies.push_back(latency);
        if (kernel_launch_latencies.size() > 1000) {
            kernel_launch_latencies.erase(kernel_launch_latencies.begin());
        }
        
        if (kernel_launch_latencies.size() < 100) {
            return result;
        }
        
        std::vector<double> sorted = kernel_launch_latencies;
        std::sort(sorted.begin(), sorted.end());
        
        double q25 = sorted[sorted.size() * 0.25];
        double q75 = sorted[sorted.size() * 0.75];
        double iqr = q75 - q25;
        
        double mean = 0.0;
        for (double l : kernel_launch_latencies) {
            mean += l;
        }
        mean /= kernel_launch_latencies.size();
        
        double cu = iqr / mean;
        
        if (cu < 0.7) {
            result.has_anomaly = true;
            result.rule_name = "R8";
            result.metric_name = "Kernel_launch_latency_uniformity";
            result.current_value = cu;
            result.threshold_value = 0.7;
            result.description = "核启动延迟分布不均匀，均匀性系数: " + std::to_string(cu);
        }
        
        return result;
    }
    
    // R9: 内存拷贝速率判断
    AnomalyResult checkMemoryCopyRate(double h2d_rate, double d2h_rate) {
        AnomalyResult result;
        
        double h2d_threshold = 5.0;
        double d2h_threshold = 3.0;
        
        if (h2d_rate < h2d_threshold) {
            result.has_anomaly = true;
            result.rule_name = "R9";
            result.metric_name = "Memory_copy_H2D";
            result.current_value = h2d_rate;
            result.threshold_value = h2d_threshold;
            result.description = "主机到设备内存拷贝速率过低: " + std::to_string(h2d_rate) + " GB/s";
            return result;
        }
        
        if (d2h_rate < d2h_threshold) {
            result.has_anomaly = true;
            result.rule_name = "R9";
            result.metric_name = "Memory_copy_D2H";
            result.current_value = d2h_rate;
            result.threshold_value = d2h_threshold;
            result.description = "设备到主机内存拷贝速率过低: " + std::to_string(d2h_rate) + " GB/s";
        }
        
        return result;
    }
    
    // R10: 步间CPU操作异常
    AnomalyResult checkInterStepCPUAnomaly(double inter_step_time, double total_step_time) {
        AnomalyResult result;
        
        if (total_step_time <= 0) {
            return result;
        }
        
        double ratio = (inter_step_time / total_step_time) * 100.0;
        
        if (ratio > 50.0) {
            result.has_anomaly = true;
            result.rule_name = "R10";
            result.metric_name = "Inter_step_CPU_ratio";
            result.current_value = ratio;
            result.threshold_value = 50.0;
            result.description = "步间CPU操作耗时占比过高: " + std::to_string(ratio) + "%";
        }
        
        return result;
    }
    
    std::vector<AnomalyResult> detectAnomalies(const Metrics& metrics) {
        std::vector<AnomalyResult> results;
        
        // R1: 静态阈值检测
        results.push_back(checkStaticThreshold("power", metrics.power));
        results.push_back(checkStaticThreshold("temperature", metrics.temperature));
        results.push_back(checkStaticThreshold("ai_core_usage", metrics.ai_core_usage));
        results.push_back(checkStaticThreshold("ai_cpu_usage", metrics.ai_cpu_usage));
        results.push_back(checkStaticThreshold("ctrl_cpu_usage", metrics.ctrl_cpu_usage));
        results.push_back(checkStaticThreshold("memory_usage", metrics.memory_usage));
        results.push_back(checkStaticThreshold("memory_bandwidth_usage", metrics.memory_bandwidth_usage));
        results.push_back(checkStaticThreshold("python_gc_time", metrics.python_gc_time));
        
        // R2: CUSUM动态趋势检测
        results.push_back(checkCUSUM("power", metrics.power));
        results.push_back(checkCUSUM("temperature", metrics.temperature));
        results.push_back(checkCUSUM("ai_core_usage", metrics.ai_core_usage));
        
        // R3: 训练吞吐量检测
        results.push_back(checkThroughput(metrics.dataloader_throughput));
        
        // R4: 计算密集型核函数FLOPS检测
        results.push_back(checkFLOPS("aclnnFlashAttentionScore", metrics.aclnnFlashAttentionScore_tflops));
        results.push_back(checkFLOPS("aclnnMatmul", metrics.aclnnMatmul_tflops));
        results.push_back(checkFLOPS("aclnnBatchMatMul", metrics.aclnnBatchMatMul_tflops));
        results.push_back(checkFLOPS("aclnnFFN", metrics.aclnnFFN_tflops));
        results.push_back(checkFLOPS("aclnnFlashAttentionScoreGrad", metrics.aclnnFlashAttentionScoreGrad_tflops));
        results.push_back(checkFLOPS("aclnnMatmul_grad", metrics.aclnnMatmul_grad_tflops));
        results.push_back(checkFLOPS("aclnnBatchMatMul_grad", metrics.aclnnBatchMatMul_grad_tflops));
        
        // R5-R6: 通信带宽检测（简化示例，假设单组单rank）
        results.push_back(checkIntraGroupAnomaly("DP0", 0, metrics.hcclAllReduce_bandwidth));
        results.push_back(checkInterGroupAnomaly("DP0", metrics.hcclAllReduce_bandwidth));
        
        // R7: 次要NPU操作检测（需要计算总step时间和监控内核时间）
        double step_time = metrics.aclnnFlashAttentionScore_time + 
                          metrics.aclnnMatmul_tflops * 0.001 +  // 简化估算
                          metrics.aclnnFFN_time;
        double monitored_time = step_time * 0.8;  // 假设监控了80%的时间
        results.push_back(checkMinorKernelAnomaly(step_time, monitored_time));
        
        // R8: 核启动延迟检测
        results.push_back(checkKernelLaunchLatency(metrics.aclrtLaunchKernel_latency));
        
        // R9: 内存拷贝速率检测（简化：从吞吐量估算）
        double h2d_rate = metrics.aclrtMemcpyAsync_throughput / 1000.0;  // MB/s to GB/s
        double d2h_rate = metrics.aclrtMemcpy2dAsync_throughput / 1000.0;
        results.push_back(checkMemoryCopyRate(h2d_rate, d2h_rate));
        
        // R10: 步间CPU操作检测（简化示例）
        double inter_step = 0.1;  // 假设步间操作时间
        double total_step = step_time + inter_step;
        results.push_back(checkInterStepCPUAnomaly(inter_step, total_step));
        
        return results;
    }
    
    void printResults(const Metrics& metrics, const std::vector<AnomalyResult>& results) {
        auto time_t = std::chrono::system_clock::to_time_t(metrics.timestamp);
        std::tm* timeinfo = std::localtime(&time_t);
        char buffer[80];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
        
        std::cout << "\n========================================\n";
        std::cout << "时间戳: " << buffer << "\n";
        std::cout << "========================================\n";
        
        std::cout << "\n【当前指标值】\n";
        std::cout << "功率: " << metrics.power << " W\n";
        std::cout << "温度: " << metrics.temperature << " ℃\n";
        std::cout << "AI Core占用率: " << metrics.ai_core_usage << " %\n";
        std::cout << "内存占用率: " << metrics.memory_usage << " %\n";
        std::cout << "DataLoader吞吐量: " << metrics.dataloader_throughput << " batch/s\n";
        std::cout << "FlashAttention TFLOPs: " << metrics.aclnnFlashAttentionScore_tflops << "\n";
        std::cout << "AllReduce带宽: " << metrics.hcclAllReduce_bandwidth << " GB/s\n";
        
        bool has_any_anomaly = false;
        for (const auto& result : results) {
            if (result.has_anomaly) {
                if (!has_any_anomaly) {
                    std::cout << "\n【⚠️  异常检测结果】\n";
                    has_any_anomaly = true;
                }
                std::cout << "⚠️  警告 [" << result.rule_name << "] " 
                         << result.metric_name << ": " << result.description << "\n";
                std::cout << "   当前值: " << result.current_value 
                         << ", 阈值: " << result.threshold_value << "\n";
            }
        }
        
        if (!has_any_anomaly) {
            std::cout << "\n【✓ 正常】所有指标均在正常范围内\n";
        }
        
        std::cout << "========================================\n\n";
    }
};

class MetricsGenerator {
private:
    double frequency_hz;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    std::normal_distribution<double> normal_dist;
    
    // 基准值（模拟正常训练场景）
    struct BaselineValues {
        double power = 250.0;              // 250W
        double temperature = 65.0;          // 65℃
        double ai_core_usage = 85.0;        // 85%
        double ai_cpu_usage = 30.0;         // 30%
        double ctrl_cpu_usage = 20.0;       // 20%
        double memory_usage = 60.0;         // 60%
        double memory_bandwidth_usage = 70.0; // 70%
        double python_gc_time = 50.0;       // 50ms
        double dataloader_throughput = 10.0; // 10 batch/s
    } baseline;
    
    int step_counter;
    
public:
    MetricsGenerator(double freq) : frequency_hz(freq), 
                                    rng(std::random_device{}()),
                                    uniform_dist(0.0, 1.0),
                                    normal_dist(0.0, 1.0),
                                    step_counter(0) {
        if (frequency_hz <= 0) {
            frequency_hz = 1.0;  // 默认1Hz
        }
    }
    
    Metrics generateMetrics() {
        Metrics m;
        m.timestamp = std::chrono::system_clock::now();
        step_counter++;
        
        double time_factor = std::sin(step_counter * 0.1) * 0.1 + 1.0;
        double noise = normal_dist(rng) * 0.1;
        
        // T1-T7: 全过程监控指标
        m.power = baseline.power * time_factor * (1.0 + noise);
        m.temperature = baseline.temperature * time_factor * (1.0 + noise);
        m.ai_core_usage = std::max(0.0, std::min(100.0, 
            baseline.ai_core_usage * time_factor * (1.0 + noise)));
        m.ai_cpu_usage = std::max(0.0, std::min(100.0, 
            baseline.ai_cpu_usage * time_factor * (1.0 + noise)));
        m.ctrl_cpu_usage = std::max(0.0, std::min(100.0, 
            baseline.ctrl_cpu_usage * time_factor * (1.0 + noise)));
        m.memory_usage = std::max(0.0, std::min(100.0, 
            baseline.memory_usage * time_factor * (1.0 + noise)));
        m.memory_bandwidth_usage = std::max(0.0, std::min(100.0, 
            baseline.memory_bandwidth_usage * time_factor * (1.0 + noise)));
        
        // T8: Python GC耗时
        m.python_gc_time = baseline.python_gc_time * (1.0 + noise * 0.5);
        
        // T9-T14: 内存相关aclrt函数吞吐量 (MB/s)
        m.aclrtMemcpyAsync_throughput = 5000.0 * (1.0 + noise);
        m.aclrtMemcpy2dAsync_throughput = 4500.0 * (1.0 + noise);
        m.aclrtFree_throughput = 3000.0 * (1.0 + noise);
        m.aclrtFreeHost_throughput = 2800.0 * (1.0 + noise);
        m.aclrtMalloc_throughput = 3200.0 * (1.0 + noise);
        m.aclrtMallocAsync_throughput = 3100.0 * (1.0 + noise);
        
        // T15: aclrtLaunchKernel启动时延 (μs)
        m.aclrtLaunchKernel_latency = 50.0 * (1.0 + noise * 0.3);
        
        // D1: DataLoader吞吐量 (batch/s)
        m.dataloader_throughput = baseline.dataloader_throughput * (1.0 + noise * 0.2);
        
        // F1: aclnnFlashAttentionScore
        double attention_time = 0.1 * (1.0 + noise * 0.15);  // 约100ms
        m.aclnnFlashAttentionScore_time = attention_time;
        // TFLOPs = 4×B×N×D×Sq×Sk/T
        // 假设: B=32, N=16, D=128, Sq=2048, Sk=2048
        double attention_tflops = 4.0 * 32 * 16 * 128 * 2048 * 2048 / (attention_time * 1e12);
        m.aclnnFlashAttentionScore_tflops = attention_tflops;
        
        // F2: aclnnMatmul
        // TFLOPs = B×M×N×2K/T
        // 假设: B=32, M=4096, N=4096, K=4096, T=0.05s
        double matmul_time = 0.05 * (1.0 + noise * 0.1);
        m.aclnnMatmul_tflops = 32.0 * 4096.0 * 4096.0 * 2.0 * 4096.0 / (matmul_time * 1e12);
        
        // F3: aclnnBatchMatMul
        double batch_matmul_time = 0.06 * (1.0 + noise * 0.1);
        m.aclnnBatchMatMul_tflops = 32.0 * 2048.0 * 2048.0 * 2.0 * 2048.0 / (batch_matmul_time * 1e12);
        
        // F4: aclnnFFN
        double ffn_time = 0.08 * (1.0 + noise * 0.1);
        m.aclnnFFN_time = ffn_time;
        // TFLOPs = (4×B×S×Din×Dmid + 3×B×S×Dmid) / T
        // 假设: B=32, S=2048, Din=4096, Dmid=16384
        m.aclnnFFN_tflops = (4.0 * 32 * 2048 * 4096 * 16384 + 3.0 * 32 * 2048 * 16384) / (ffn_time * 1e12);
        
        // B1: aclnnFlashAttentionScoreGrad
        double grad_attention_time = 0.12 * (1.0 + noise * 0.15);
        // TFLOPs = B×N×8×Sq×Sk×D/T
        m.aclnnFlashAttentionScoreGrad_tflops = 32.0 * 16.0 * 8.0 * 2048.0 * 2048.0 * 128.0 / (grad_attention_time * 1e12);
        
        // B2-B3: 矩阵乘法梯度
        double grad_matmul_time = 0.06 * (1.0 + noise * 0.1);
        m.aclnnMatmul_grad_tflops = 32.0 * 4096.0 * 4096.0 * 2.0 * 4096.0 / (grad_matmul_time * 1e12);
        m.aclnnBatchMatMul_grad_tflops = 32.0 * 2048.0 * 2048.0 * 2.0 * 2048.0 / (grad_matmul_time * 1e12);
        
        // B4-B5: torch.autograd
        m.torch_autograd_backward_time = 500.0 * (1.0 + noise * 0.2);  // ms
        m.torch_autograd_grad_time = 200.0 * (1.0 + noise * 0.15);     // ms
        
        // G1-G4: hccl通信带宽 (GB/s)
        m.hcclAllReduce_bandwidth = 25.0 * (1.0 + noise * 0.1);
        m.hcclBroadcast_bandwidth = 30.0 * (1.0 + noise * 0.1);
        m.hcclAllGather_bandwidth = 28.0 * (1.0 + noise * 0.1);
        m.hcclReduceScatter_bandwidth = 27.0 * (1.0 + noise * 0.1);
        
        // G5-G7: 同步耗时 (μs)
        m.aclrtSynchronizeStream_time = 100.0 * (1.0 + noise * 0.2);
        m.aclrtSynchronizeEvent_time = 50.0 * (1.0 + noise * 0.2);
        m.aclrtStreamWaitEvent_time = 80.0 * (1.0 + noise * 0.2);
        
        return m;
    }
    
    double getFrequency() const { return frequency_hz; }
};

int main(int argc, char* argv[]) {
    double frequency = 1.0;
    
    if (argc > 1) {
        frequency = std::stod(argv[1]);
        if (frequency <= 0) {
            std::cerr << "错误：频率必须大于0，使用默认值1.0 Hz\n";
            frequency = 1.0;
        }
    }
    
    std::cout << "========================================\n";
    std::cout << "  昇腾NPU分布式训练监测系统\n";
    std::cout << "  指标生成频率: " << frequency << " Hz\n";
    std::cout << "========================================\n\n";
    
    MetricsGenerator generator(frequency);
    AnomalyDetector detector;
    
    std::chrono::milliseconds interval(
        static_cast<int>(1000.0 / frequency)
    );
    
    std::cout << "系统已启动，开始生成指标并进行异常检测...\n";
    std::cout << "按 Ctrl+C 停止运行\n\n";
    
    while (true) {
        Metrics metrics = generator.generateMetrics();
        
        auto results = detector.detectAnomalies(metrics);
        
        detector.printResults(metrics, results);
        
        std::this_thread::sleep_for(interval);
    }
    
    return 0;
}

