#include "ProtocolHandler.h"
#include "monitor.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <cstring>

ProtocolHandler::ProtocolHandler() {
}

ProtocolHandler::~ProtocolHandler() {
}

bool ProtocolHandler::ParseMessage(const char* data, size_t length, monitor::MonitorData& output) {
    if (data == nullptr || length == 0) {
        return false;
    }
    
    // Protobuf消息格式：4字节长度前缀 + 消息体
    if (length < 4) {
        return false;
    }
    
    // 读取长度前缀
    uint32_t message_length = 0;
    memcpy(&message_length, data, 4);
    message_length = ntohl(message_length); // 网络字节序转换
    
    if (message_length > MAX_MESSAGE_SIZE || message_length > length - 4) {
        std::cerr << "Invalid message length: " << message_length << std::endl;
        return false;
    }
    
    // 解析Protobuf消息
    const char* message_data = data + 4;
    google::protobuf::io::ArrayInputStream input_stream(message_data, message_length);
    google::protobuf::io::CodedInputStream coded_stream(&input_stream);
    
    if (!output.ParseFromCodedStream(&coded_stream)) {
        std::cerr << "Failed to parse MonitorData message" << std::endl;
        return false;
    }
    
    return ValidateMessage(output);
}

bool ProtocolHandler::ParseBatchMessage(const char* data, size_t length, monitor::BatchMonitorData& output) {
    if (data == nullptr || length == 0) {
        return false;
    }
    
    // Protobuf消息格式：4字节长度前缀 + 消息体
    if (length < 4) {
        return false;
    }
    
    // 读取长度前缀
    uint32_t message_length = 0;
    memcpy(&message_length, data, 4);
    message_length = ntohl(message_length);
    
    if (message_length > MAX_MESSAGE_SIZE || message_length > length - 4) {
        std::cerr << "Invalid batch message length: " << message_length << std::endl;
        return false;
    }
    
    // 解析Protobuf消息
    const char* message_data = data + 4;
    google::protobuf::io::ArrayInputStream input_stream(message_data, message_length);
    google::protobuf::io::CodedInputStream coded_stream(&input_stream);
    
    if (!output.ParseFromCodedStream(&coded_stream)) {
        std::cerr << "Failed to parse BatchMonitorData message" << std::endl;
        return false;
    }
    
    return true;
}

bool ProtocolHandler::ValidateMessage(const monitor::MonitorData& data) {
    // 验证必要字段
    if (data.node_id().empty()) {
        std::cerr << "Missing node_id" << std::endl;
        return false;
    }
    
    if (data.metric_name().empty()) {
        std::cerr << "Missing metric_name" << std::endl;
        return false;
    }
    
    if (data.timestamp() <= 0) {
        std::cerr << "Invalid timestamp: " << data.timestamp() << std::endl;
        return false;
    }
    
    return true;
}

int ProtocolHandler::GetProtocolVersion(const char* data, size_t length) {
    // 简化实现：返回版本1
    // 实际实现中可以从消息头中读取版本号
    (void)data;
    (void)length;
    return 1;
}

bool ProtocolHandler::IsMessageComplete(const char* data, size_t length, size_t& message_length) {
    if (length < 4) {
        return false; // 长度前缀不完整
    }
    
    uint32_t msg_len = 0;
    memcpy(&msg_len, data, 4);
    msg_len = ntohl(msg_len);
    
    if (msg_len > MAX_MESSAGE_SIZE) {
        return false; // 消息长度超出限制
    }
    
    message_length = msg_len + 4; // 包括4字节长度前缀
    
    return length >= message_length; // 检查是否收到完整消息
}

