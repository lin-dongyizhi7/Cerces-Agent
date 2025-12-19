#include "DataTransformer.h"
#include "monitor.pb.h"
#include <regex>
#include <algorithm>

DataTransformer::DataTransformer() {
}

DataTransformer::~DataTransformer() {
}

StructuredData DataTransformer::Transform(const monitor::MonitorData& proto_data) {
    StructuredData structured_data;
    
    structured_data.node_id = proto_data.node_id();
    structured_data.rank_id = ExtractRankId(proto_data.tags());
    structured_data.timestamp_us = proto_data.timestamp();
    structured_data.metric_name = proto_data.metric_name();
    structured_data.value = proto_data.value();
    structured_data.unit = proto_data.unit();
    structured_data.step_id = ExtractStepId(proto_data.tags());
    structured_data.metric_type = InferMetricType(proto_data.metric_name());
    
    // 复制tags到metadata
    for (const auto& pair : proto_data.tags()) {
        structured_data.metadata[pair.first] = pair.second;
    }
    
    // 数据标准化
    Normalize(structured_data);
    
    return structured_data;
}

std::vector<StructuredData> DataTransformer::TransformBatch(const monitor::BatchMonitorData& batch_data) {
    std::vector<StructuredData> result;
    result.reserve(batch_data.data_size());
    
    for (const auto& proto_data : batch_data.data()) {
        result.push_back(Transform(proto_data));
    }
    
    return result;
}

void DataTransformer::Normalize(StructuredData& data) {
    // 标准化时间戳（确保是微秒）
    // 如果时间戳是秒，转换为微秒
    if (data.timestamp_us < 1000000000000LL) { // 假设小于这个值的是秒级时间戳
        data.timestamp_us *= 1000000;
    }
    
    // 标准化单位
    if (data.unit.empty()) {
        // 根据指标名称推断单位
        if (data.metric_name.find("temperature") != std::string::npos ||
            data.metric_name.find("temp") != std::string::npos) {
            data.unit = "°C";
        } else if (data.metric_name.find("power") != std::string::npos) {
            data.unit = "W";
        } else if (data.metric_name.find("rate") != std::string::npos ||
                   data.metric_name.find("throughput") != std::string::npos) {
            data.unit = "ops/s";
        } else if (data.metric_name.find("usage") != std::string::npos ||
                   data.metric_name.find("occupancy") != std::string::npos) {
            data.unit = "%";
        }
    }
}

std::string DataTransformer::ExtractRankId(const std::map<std::string, std::string>& tags) {
    auto it = tags.find("rank_id");
    if (it != tags.end()) {
        return it->second;
    }
    
    // 尝试其他可能的键名
    for (const auto& pair : tags) {
        std::string key = pair.first;
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        if (key.find("rank") != std::string::npos) {
            return pair.second;
        }
    }
    
    return "";
}

int32_t DataTransformer::ExtractStepId(const std::map<std::string, std::string>& tags) {
    auto it = tags.find("step_id");
    if (it != tags.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {
            return -1;
        }
    }
    
    // 尝试其他可能的键名
    for (const auto& pair : tags) {
        std::string key = pair.first;
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        if (key.find("step") != std::string::npos) {
            try {
                return std::stoi(pair.second);
            } catch (...) {
                return -1;
            }
        }
    }
    
    return -1;
}

std::string DataTransformer::InferMetricType(const std::string& metric_name) {
    // 根据指标名称推断类型
    std::string lower_name = metric_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    // T系列指标（全过程监控指标）
    if (lower_name.find("power") != std::string::npos) return "T1";
    if (lower_name.find("temperature") != std::string::npos || lower_name.find("temp") != std::string::npos) return "T2";
    if (lower_name.find("ai_core") != std::string::npos && lower_name.find("usage") != std::string::npos) return "T3";
    if (lower_name.find("ai_cpu") != std::string::npos && lower_name.find("usage") != std::string::npos) return "T4";
    if (lower_name.find("ctrl_cpu") != std::string::npos && lower_name.find("usage") != std::string::npos) return "T5";
    if (lower_name.find("memory") != std::string::npos && lower_name.find("usage") != std::string::npos) return "T6";
    if (lower_name.find("memory") != std::string::npos && lower_name.find("bandwidth") != std::string::npos) return "T7";
    
    // D系列指标（DataLoader）
    if (lower_name.find("dataloader") != std::string::npos) return "D1";
    
    // F系列指标（前向传播）
    if (lower_name.find("flashattention") != std::string::npos) return "F1";
    if (lower_name.find("matmul") != std::string::npos && lower_name.find("forward") != std::string::npos) return "F2";
    if (lower_name.find("batchmatmul") != std::string::npos && lower_name.find("forward") != std::string::npos) return "F3";
    if (lower_name.find("ffn") != std::string::npos && lower_name.find("forward") != std::string::npos) return "F4";
    
    // B系列指标（反向传播）
    if (lower_name.find("flashattention") != std::string::npos && lower_name.find("grad") != std::string::npos) return "B1";
    if (lower_name.find("matmul") != std::string::npos && lower_name.find("backward") != std::string::npos) return "B2";
    if (lower_name.find("batchmatmul") != std::string::npos && lower_name.find("backward") != std::string::npos) return "B3";
    if (lower_name.find("autograd") != std::string::npos && lower_name.find("backward") != std::string::npos) return "B4";
    if (lower_name.find("autograd") != std::string::npos && lower_name.find("grad") != std::string::npos) return "B5";
    
    // G系列指标（梯度同步和参数更新）
    if (lower_name.find("hccl") != std::string::npos) {
        if (lower_name.find("allreduce") != std::string::npos) return "G1";
        if (lower_name.find("broadcast") != std::string::npos) return "G2";
        if (lower_name.find("allgather") != std::string::npos) return "G3";
        if (lower_name.find("reducescatter") != std::string::npos) return "G4";
    }
    if (lower_name.find("cann") != std::string::npos && lower_name.find("sync") != std::string::npos) return "G5";
    
    // 默认返回空字符串
    return "";
}

