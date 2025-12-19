#ifndef PROTOCOL_HANDLER_H
#define PROTOCOL_HANDLER_H

#include <string>
#include <cstddef>
#include "StructuredData.h"

// 前向声明Protobuf消息
namespace monitor {
    class MonitorData;
    class BatchMonitorData;
}

class ProtocolHandler {
public:
    ProtocolHandler();
    ~ProtocolHandler();

    // 解析Protobuf消息（单条）
    bool ParseMessage(const char* data, size_t length, monitor::MonitorData& output);
    
    // 解析批量Protobuf消息
    bool ParseBatchMessage(const char* data, size_t length, monitor::BatchMonitorData& output);
    
    // 验证消息完整性
    bool ValidateMessage(const monitor::MonitorData& data);
    
    // 获取协议版本
    int GetProtocolVersion(const char* data, size_t length);
    
    // 检查消息是否完整（检查长度前缀）
    bool IsMessageComplete(const char* data, size_t length, size_t& message_length);

private:
    static const int MAX_MESSAGE_SIZE = 10485760; // 10MB
};

#endif // PROTOCOL_HANDLER_H

