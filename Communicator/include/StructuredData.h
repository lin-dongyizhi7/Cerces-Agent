#ifndef STRUCTURED_DATA_H
#define STRUCTURED_DATA_H

#include <string>
#include <map>
#include <cstdint>

// 结构化数据格式
struct StructuredData {
    std::string node_id;
    std::string rank_id;
    int64_t timestamp_us;      // 微秒时间戳
    std::string metric_type;   // 指标类型（T1-T15, D1, F1-F4, B1-B5, G1-G7）
    std::string metric_name;   // 指标名称
    double value;              // 指标值
    std::string unit;          // 单位
    int32_t step_id;           // 训练步数
    std::map<std::string, std::string> metadata; // 元数据
};

#endif // STRUCTURED_DATA_H

