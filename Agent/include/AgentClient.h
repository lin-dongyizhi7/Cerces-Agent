#ifndef AGENT_CLIENT_H
#define AGENT_CLIENT_H

#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include "MetricGenerator.h"

// Agent客户端 - 负责连接Communicator并发送数据
class AgentClient {
public:
    AgentClient(const std::string& server_host, int server_port,
                const std::string& node_id, const std::string& rank_id);
    ~AgentClient();
    
    // 启动客户端
    bool Start();
    
    // 停止客户端
    void Stop();
    
    // 发送单条指标数据
    bool SendMetric(const MetricGenerator::MetricData& metric);
    
    // 批量发送指标数据
    bool SendMetrics(const std::vector<MetricGenerator::MetricData>& metrics);
    
    // 启动流式发送（自动生成并发送数据）
    void StartStreaming(int interval_ms = 1000);
    
    // 停止流式发送
    void StopStreaming();
    
    // 检查连接状态
    bool IsConnected() const;
    
    // 设置发送间隔（毫秒）
    void SetSendInterval(int interval_ms);

private:
    std::string server_host_;
    int server_port_;
    std::string node_id_;
    std::string rank_id_;
    
    int socket_fd_;
    std::atomic<bool> is_connected_;
    std::atomic<bool> is_streaming_;
    
    std::thread streaming_thread_;
    int send_interval_ms_;
    
    MetricGenerator metric_generator_;
    
    // 内部方法
    bool Connect();
    void Disconnect();
    bool SendData(const char* data, size_t length);
    void StreamingLoop();
    
    // 序列化指标数据为Protobuf格式
    bool SerializeMetric(const MetricGenerator::MetricData& metric, 
                        std::vector<char>& buffer);
};

#endif // AGENT_CLIENT_H

