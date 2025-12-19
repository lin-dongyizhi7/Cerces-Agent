#ifndef DATA_TRANSFORMER_H
#define DATA_TRANSFORMER_H

#include <vector>
#include "StructuredData.h"

// 前向声明Protobuf消息
namespace monitor {
    class MonitorData;
    class BatchMonitorData;
}

class DataTransformer {
public:
    DataTransformer();
    ~DataTransformer();

    // 转换单条数据
    StructuredData Transform(const monitor::MonitorData& proto_data);
    
    // 批量转换
    std::vector<StructuredData> TransformBatch(const monitor::BatchMonitorData& batch_data);
    
    // 数据标准化
    void Normalize(StructuredData& data);

private:
    // 从tags中提取rank_id
    std::string ExtractRankId(const std::map<std::string, std::string>& tags);
    
    // 从tags中提取step_id
    int32_t ExtractStepId(const std::map<std::string, std::string>& tags);
    
    // 从metric_name推断metric_type
    std::string InferMetricType(const std::string& metric_name);
};

#endif // DATA_TRANSFORMER_H

