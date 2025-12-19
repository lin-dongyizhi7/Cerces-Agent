#ifndef METRIC_GENERATOR_H
#define METRIC_GENERATOR_H

#include <string>
#include <cstdint>
#include <random>
#include <map>

// 指标数据生成器
class MetricGenerator {
public:
    MetricGenerator(const std::string& node_id, const std::string& rank_id);
    ~MetricGenerator();
    
    // 生成单条指标数据
    struct MetricData {
        std::string metric_name;
        double value;
        std::string unit;
        std::string metric_type;
        int32_t step_id;
    };
    
    MetricData GenerateMetric(const std::string& metric_name);
    
    // 生成指定类型的指标
    MetricData GenerateTemperature();      // T2
    MetricData GeneratePower();            // T1
    MetricData GenerateAICoreUsage();      // T3
    MetricData GenerateMemoryUsage();      // T6
    MetricData GenerateDataLoader();       // D1
    MetricData GenerateMatmul();           // F2
    MetricData GenerateHCCLAllReduce();    // G1
    
    // 生成随机指标（用于测试）
    MetricData GenerateRandomMetric();
    
    // 设置当前步数
    void SetStepId(int32_t step_id);
    
    // 获取当前时间戳（微秒）
    int64_t GetCurrentTimestamp() const;

private:
    std::string node_id_;
    std::string rank_id_;
    int32_t current_step_id_;
    
    // 随机数生成器
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    
    // 指标值的范围（用于生成合理的模拟数据）
    struct MetricRange {
        double min;
        double max;
        double base;  // 基准值
    };
    std::map<std::string, MetricRange> metric_ranges_;
    
    void InitializeMetricRanges();
    double GenerateValueWithNoise(double base, double noise_level = 0.1);
};

#endif // METRIC_GENERATOR_H

