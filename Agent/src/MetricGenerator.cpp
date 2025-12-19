#include "MetricGenerator.h"
#include <chrono>
#include <cmath>
#include <algorithm>

MetricGenerator::MetricGenerator(const std::string& node_id, const std::string& rank_id)
    : node_id_(node_id), rank_id_(rank_id), current_step_id_(0),
      rng_(std::random_device{}()), uniform_dist_(0.0, 1.0) {
    InitializeMetricRanges();
}

MetricGenerator::~MetricGenerator() {
}

void MetricGenerator::InitializeMetricRanges() {
    // 初始化各指标的合理范围
    metric_ranges_["T1_power"] = {0.0, 300.0, 150.0};           // 功率 (W)
    metric_ranges_["T2_temperature"] = {0.0, 85.0, 45.0};      // 温度 (°C)
    metric_ranges_["T3_ai_core_usage"] = {0.0, 100.0, 60.0};   // AI Core占用率 (%)
    metric_ranges_["T4_ai_cpu_usage"] = {0.0, 100.0, 30.0};    // AI CPU占用率 (%)
    metric_ranges_["T5_ctrl_cpu_usage"] = {0.0, 100.0, 20.0};   // Ctrl CPU占用率 (%)
    metric_ranges_["T6_memory_usage"] = {0.0, 100.0, 50.0};    // 内存占用率 (%)
    metric_ranges_["T7_memory_bandwidth"] = {0.0, 100.0, 40.0}; // 内存带宽占用率 (%)
    metric_ranges_["D1_dataloader"] = {0.0, 10000.0, 5000.0};  // DataLoader吞吐量 (ops/s)
    metric_ranges_["F2_matmul"] = {0.0, 100000.0, 50000.0};    // Matmul FLOPS
    metric_ranges_["G1_hccl_allreduce"] = {0.0, 10000.0, 5000.0}; // AllReduce带宽 (MB/s)
}

MetricGenerator::MetricData MetricGenerator::GenerateMetric(const std::string& metric_name) {
    MetricData data;
    data.metric_name = metric_name;
    data.step_id = current_step_id_;
    
    // 根据指标名称生成对应的值
    if (metric_name.find("temperature") != std::string::npos || 
        metric_name.find("temp") != std::string::npos) {
        return GenerateTemperature();
    } else if (metric_name.find("power") != std::string::npos) {
        return GeneratePower();
    } else if (metric_name.find("ai_core") != std::string::npos && 
               metric_name.find("usage") != std::string::npos) {
        return GenerateAICoreUsage();
    } else if (metric_name.find("memory") != std::string::npos && 
               metric_name.find("usage") != std::string::npos) {
        return GenerateMemoryUsage();
    } else if (metric_name.find("dataloader") != std::string::npos) {
        return GenerateDataLoader();
    } else if (metric_name.find("matmul") != std::string::npos) {
        return GenerateMatmul();
    } else if (metric_name.find("hccl") != std::string::npos && 
               metric_name.find("allreduce") != std::string::npos) {
        return GenerateHCCLAllReduce();
    } else {
        return GenerateRandomMetric();
    }
}

MetricGenerator::MetricData MetricGenerator::GenerateTemperature() {
    MetricData data;
    data.metric_name = "temperature";
    data.metric_type = "T2";
    data.unit = "°C";
    data.step_id = current_step_id_;
    
    // 生成温度值：基准值 + 随机波动 + 可能的步数相关变化
    double base = metric_ranges_["T2_temperature"].base;
    double value = GenerateValueWithNoise(base, 0.15);
    value = std::max(metric_ranges_["T2_temperature"].min, 
                     std::min(metric_ranges_["T2_temperature"].max, value));
    data.value = value;
    
    return data;
}

MetricGenerator::MetricData MetricGenerator::GeneratePower() {
    MetricData data;
    data.metric_name = "power";
    data.metric_type = "T1";
    data.unit = "W";
    data.step_id = current_step_id_;
    
    double base = metric_ranges_["T1_power"].base;
    double value = GenerateValueWithNoise(base, 0.2);
    value = std::max(metric_ranges_["T1_power"].min, 
                     std::min(metric_ranges_["T1_power"].max, value));
    data.value = value;
    
    return data;
}

MetricGenerator::MetricData MetricGenerator::GenerateAICoreUsage() {
    MetricData data;
    data.metric_name = "ai_core_usage";
    data.metric_type = "T3";
    data.unit = "%";
    data.step_id = current_step_id_;
    
    double base = metric_ranges_["T3_ai_core_usage"].base;
    double value = GenerateValueWithNoise(base, 0.25);
    value = std::max(metric_ranges_["T3_ai_core_usage"].min, 
                     std::min(metric_ranges_["T3_ai_core_usage"].max, value));
    data.value = value;
    
    return data;
}

MetricGenerator::MetricData MetricGenerator::GenerateMemoryUsage() {
    MetricData data;
    data.metric_name = "memory_usage";
    data.metric_type = "T6";
    data.unit = "%";
    data.step_id = current_step_id_;
    
    double base = metric_ranges_["T6_memory_usage"].base;
    double value = GenerateValueWithNoise(base, 0.2);
    value = std::max(metric_ranges_["T6_memory_usage"].min, 
                     std::min(metric_ranges_["T6_memory_usage"].max, value));
    data.value = value;
    
    return data;
}

MetricGenerator::MetricData MetricGenerator::GenerateDataLoader() {
    MetricData data;
    data.metric_name = "dataloader_throughput";
    data.metric_type = "D1";
    data.unit = "ops/s";
    data.step_id = current_step_id_;
    
    double base = metric_ranges_["D1_dataloader"].base;
    double value = GenerateValueWithNoise(base, 0.3);
    value = std::max(metric_ranges_["D1_dataloader"].min, 
                     std::min(metric_ranges_["D1_dataloader"].max, value));
    data.value = value;
    
    return data;
}

MetricGenerator::MetricData MetricGenerator::GenerateMatmul() {
    MetricData data;
    data.metric_name = "aclnnMatmul_flops";
    data.metric_type = "F2";
    data.unit = "FLOPS";
    data.step_id = current_step_id_;
    
    double base = metric_ranges_["F2_matmul"].base;
    double value = GenerateValueWithNoise(base, 0.25);
    value = std::max(metric_ranges_["F2_matmul"].min, 
                     std::min(metric_ranges_["F2_matmul"].max, value));
    data.value = value;
    
    return data;
}

MetricGenerator::MetricData MetricGenerator::GenerateHCCLAllReduce() {
    MetricData data;
    data.metric_name = "hcclAllReduce_bandwidth";
    data.metric_type = "G1";
    data.unit = "MB/s";
    data.step_id = current_step_id_;
    
    double base = metric_ranges_["G1_hccl_allreduce"].base;
    double value = GenerateValueWithNoise(base, 0.3);
    value = std::max(metric_ranges_["G1_hccl_allreduce"].min, 
                     std::min(metric_ranges_["G1_hccl_allreduce"].max, value));
    data.value = value;
    
    return data;
}

MetricGenerator::MetricData MetricGenerator::GenerateRandomMetric() {
    MetricData data;
    data.metric_name = "random_metric";
    data.metric_type = "UNKNOWN";
    data.unit = "";
    data.step_id = current_step_id_;
    data.value = uniform_dist_(rng_) * 100.0;
    
    return data;
}

void MetricGenerator::SetStepId(int32_t step_id) {
    current_step_id_ = step_id;
}

int64_t MetricGenerator::GetCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

double MetricGenerator::GenerateValueWithNoise(double base, double noise_level) {
    // 生成带噪声的值：base * (1 + noise_level * random(-1, 1))
    double noise = (uniform_dist_(rng_) * 2.0 - 1.0) * noise_level;
    return base * (1.0 + noise);
}

