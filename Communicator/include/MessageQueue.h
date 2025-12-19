#ifndef MESSAGE_QUEUE_H
#define MESSAGE_QUEUE_H

#include <vector>
#include <string>
#include "StructuredData.h"

// 队列状态
struct QueueStatus {
    size_t detection_queue_size;
    size_t visualization_queue_size;
    bool is_connected;
};

class MessageQueue {
public:
    MessageQueue();
    ~MessageQueue();

    // 初始化消息队列
    bool Initialize(const std::string& detection_endpoint, 
                    const std::string& visualization_endpoint);
    
    // 发送数据到异常检测层
    bool SendToDetection(const StructuredData& data);
    
    // 发送数据到可视化层
    bool SendToVisualization(const StructuredData& data);
    
    // 批量发送
    bool SendBatch(const std::vector<StructuredData>& data_list);
    
    // 获取队列状态
    QueueStatus GetStatus();
    
    // 关闭消息队列
    void Close();

private:
    class MessageQueueImpl;
    std::unique_ptr<MessageQueueImpl> impl_;
};

#endif // MESSAGE_QUEUE_H

