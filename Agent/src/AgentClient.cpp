#include "AgentClient.h"
#include "monitor.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <chrono>
#include <vector>

AgentClient::AgentClient(const std::string& server_host, int server_port,
                         const std::string& node_id, const std::string& rank_id)
    : server_host_(server_host), server_port_(server_port),
      node_id_(node_id), rank_id_(rank_id),
      socket_fd_(-1), is_connected_(false), is_streaming_(false),
      send_interval_ms_(1000), metric_generator_(node_id, rank_id) {
}

AgentClient::~AgentClient() {
    Stop();
}

bool AgentClient::Connect() {
    if (is_connected_) {
        return true;
    }
    
    // 创建socket
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ < 0) {
        std::cerr << "Failed to create socket: " << strerror(errno) << std::endl;
        return false;
    }
    
    // 设置服务器地址
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port_);
    
    if (inet_pton(AF_INET, server_host_.c_str(), &server_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address: " << server_host_ << std::endl;
        close(socket_fd_);
        socket_fd_ = -1;
        return false;
    }
    
    // 连接服务器
    if (connect(socket_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Failed to connect to " << server_host_ << ":" << server_port_
                  << ": " << strerror(errno) << std::endl;
        close(socket_fd_);
        socket_fd_ = -1;
        return false;
    }
    
    is_connected_ = true;
    std::cout << "Connected to server " << server_host_ << ":" << server_port_ << std::endl;
    return true;
}

void AgentClient::Disconnect() {
    if (socket_fd_ >= 0) {
        close(socket_fd_);
        socket_fd_ = -1;
    }
    is_connected_ = false;
}

bool AgentClient::SendData(const char* data, size_t length) {
    if (!is_connected_ || socket_fd_ < 0) {
        return false;
    }
    
    ssize_t sent = send(socket_fd_, data, length, 0);
    if (sent < 0 || static_cast<size_t>(sent) != length) {
        std::cerr << "Failed to send data: " << strerror(errno) << std::endl;
        Disconnect();
        return false;
    }
    
    return true;
}

bool AgentClient::SerializeMetric(const MetricGenerator::MetricData& metric, 
                                  std::vector<char>& buffer) {
    // 创建Protobuf消息
    monitor::MonitorData proto_data;
    proto_data.set_node_id(node_id_);
    proto_data.set_timestamp(metric_generator_.GetCurrentTimestamp());
    proto_data.set_metric_name(metric.metric_name);
    proto_data.set_value(metric.value);
    proto_data.set_unit(metric.unit);
    
    // 添加tags
    proto_data.mutable_tags()->insert({"rank_id", rank_id_});
    proto_data.mutable_tags()->insert({"step_id", std::to_string(metric.step_id)});
    if (!metric.metric_type.empty()) {
        proto_data.mutable_tags()->insert({"metric_type", metric.metric_type});
    }
    
    // 序列化消息
    std::string serialized;
    if (!proto_data.SerializeToString(&serialized)) {
        std::cerr << "Failed to serialize Protobuf message" << std::endl;
        return false;
    }
    
    // 添加4字节长度前缀（网络字节序）
    uint32_t length = htonl(static_cast<uint32_t>(serialized.size()));
    
    buffer.resize(4 + serialized.size());
    memcpy(buffer.data(), &length, 4);
    memcpy(buffer.data() + 4, serialized.data(), serialized.size());
    
    return true;
}

bool AgentClient::Start() {
    return Connect();
}

void AgentClient::Stop() {
    StopStreaming();
    Disconnect();
}

bool AgentClient::SendMetric(const MetricGenerator::MetricData& metric) {
    if (!is_connected_ && !Connect()) {
        return false;
    }
    
    std::vector<char> buffer;
    if (!SerializeMetric(metric, buffer)) {
        return false;
    }
    
    return SendData(buffer.data(), buffer.size());
}

bool AgentClient::SendMetrics(const std::vector<MetricGenerator::MetricData>& metrics) {
    if (metrics.empty()) {
        return true;
    }
    
    if (!is_connected_ && !Connect()) {
        return false;
    }
    
    // 创建批量消息
    monitor::BatchMonitorData batch_data;
    
    for (const auto& metric : metrics) {
        auto* proto_data = batch_data.add_data();
        proto_data->set_node_id(node_id_);
        proto_data->set_timestamp(metric_generator_.GetCurrentTimestamp());
        proto_data->set_metric_name(metric.metric_name);
        proto_data->set_value(metric.value);
        proto_data->set_unit(metric.unit);
        
        proto_data->mutable_tags()->insert({"rank_id", rank_id_});
        proto_data->mutable_tags()->insert({"step_id", std::to_string(metric.step_id)});
        if (!metric.metric_type.empty()) {
            proto_data->mutable_tags()->insert({"metric_type", metric.metric_type});
        }
    }
    
    // 序列化批量消息
    std::string serialized;
    if (!batch_data.SerializeToString(&serialized)) {
        std::cerr << "Failed to serialize BatchMonitorData" << std::endl;
        return false;
    }
    
    // 添加长度前缀
    uint32_t length = htonl(static_cast<uint32_t>(serialized.size()));
    
    std::vector<char> buffer(4 + serialized.size());
    memcpy(buffer.data(), &length, 4);
    memcpy(buffer.data() + 4, serialized.data(), serialized.size());
    
    return SendData(buffer.data(), buffer.size());
}

void AgentClient::StartStreaming(int interval_ms) {
    if (is_streaming_) {
        return;
    }
    
    send_interval_ms_ = interval_ms;
    is_streaming_ = true;
    
    if (!is_connected_ && !Connect()) {
        std::cerr << "Failed to connect, cannot start streaming" << std::endl;
        is_streaming_ = false;
        return;
    }
    
    streaming_thread_ = std::thread(&AgentClient::StreamingLoop, this);
    std::cout << "Started streaming metrics every " << interval_ms << "ms" << std::endl;
}

void AgentClient::StopStreaming() {
    if (!is_streaming_) {
        return;
    }
    
    is_streaming_ = false;
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }
    std::cout << "Stopped streaming" << std::endl;
}

void AgentClient::StreamingLoop() {
    int step_id = 0;
    std::vector<std::string> metric_names = {
        "temperature", "power", "ai_core_usage", "memory_usage",
        "dataloader_throughput", "aclnnMatmul_flops", "hcclAllReduce_bandwidth"
    };
    
    while (is_streaming_) {
        metric_generator_.SetStepId(step_id);
        
        // 生成多条指标数据
        std::vector<MetricGenerator::MetricData> metrics;
        for (const auto& name : metric_names) {
            metrics.push_back(metric_generator_.GenerateMetric(name));
        }
        
        // 批量发送
        if (!SendMetrics(metrics)) {
            // 发送失败，尝试重连
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            if (!Connect()) {
                std::cerr << "Failed to reconnect, stopping stream" << std::endl;
                break;
            }
            continue;
        }
        
        step_id++;
        
        // 等待指定间隔
        std::this_thread::sleep_for(std::chrono::milliseconds(send_interval_ms_));
    }
}

bool AgentClient::IsConnected() const {
    return is_connected_;
}

void AgentClient::SetSendInterval(int interval_ms) {
    send_interval_ms_ = interval_ms;
}

