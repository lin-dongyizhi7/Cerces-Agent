#include "MessageQueue.h"
#include <iostream>
#include <sstream>
#include <zmq.hpp>
#include <mutex>
#include <thread>
#include <chrono>

class MessageQueue::MessageQueueImpl {
public:
    MessageQueueImpl() : context_(1), detection_socket_(nullptr), 
                        visualization_socket_(nullptr), is_initialized_(false) {}
    
    ~MessageQueueImpl() {
        Close();
    }
    
    bool Initialize(const std::string& detection_endpoint, 
                   const std::string& visualization_endpoint) {
        try {
            // 创建PUSH socket用于发送到异常检测层
            detection_socket_ = std::make_unique<zmq::socket_t>(context_, ZMQ_PUSH);
            detection_socket_->connect(detection_endpoint);
            
            // 创建PUB socket用于发布到可视化层
            visualization_socket_ = std::make_unique<zmq::socket_t>(context_, ZMQ_PUB);
            visualization_socket_->bind(visualization_endpoint);
            
            // 等待连接建立
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            is_initialized_ = true;
            std::cout << "MessageQueue initialized: detection=" << detection_endpoint 
                      << ", visualization=" << visualization_endpoint << std::endl;
            return true;
        } catch (const zmq::error_t& e) {
            std::cerr << "Failed to initialize MessageQueue: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool SendToDetection(const StructuredData& data) {
        if (!is_initialized_ || !detection_socket_) {
            return false;
        }
        
        try {
            // 序列化数据为JSON格式（简化实现，实际可以使用更高效的格式）
            std::string json = SerializeToJson(data);
            
            zmq::message_t message(json.size());
            memcpy(message.data(), json.c_str(), json.size());
            
            detection_socket_->send(message, zmq::send_flags::dontwait);
            return true;
        } catch (const zmq::error_t& e) {
            std::cerr << "Failed to send to detection: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool SendToVisualization(const StructuredData& data) {
        if (!is_initialized_ || !visualization_socket_) {
            return false;
        }
        
        try {
            // 序列化数据为JSON格式
            std::string json = SerializeToJson(data);
            
            zmq::message_t message(json.size());
            memcpy(message.data(), json.c_str(), json.size());
            
            visualization_socket_->send(message, zmq::send_flags::dontwait);
            return true;
        } catch (const zmq::error_t& e) {
            std::cerr << "Failed to send to visualization: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool SendBatch(const std::vector<StructuredData>& data_list) {
        bool all_success = true;
        for (const auto& data : data_list) {
            if (!SendToDetection(data)) {
                all_success = false;
            }
            if (!SendToVisualization(data)) {
                all_success = false;
            }
        }
        return all_success;
    }
    
    QueueStatus GetStatus() {
        QueueStatus status;
        status.is_connected = is_initialized_;
        // 简化实现：ZeroMQ不直接提供队列大小，这里返回固定值
        status.detection_queue_size = 0;
        status.visualization_queue_size = 0;
        return status;
    }
    
    void Close() {
        is_initialized_ = false;
        detection_socket_.reset();
        visualization_socket_.reset();
    }
    
private:
    std::string SerializeToJson(const StructuredData& data) {
        std::ostringstream oss;
        oss << "{"
            << "\"node_id\":\"" << data.node_id << "\","
            << "\"rank_id\":\"" << data.rank_id << "\","
            << "\"timestamp_us\":" << data.timestamp_us << ","
            << "\"metric_type\":\"" << data.metric_type << "\","
            << "\"metric_name\":\"" << data.metric_name << "\","
            << "\"value\":" << data.value << ","
            << "\"unit\":\"" << data.unit << "\","
            << "\"step_id\":" << data.step_id;
        
        if (!data.metadata.empty()) {
            oss << ",\"metadata\":{";
            bool first = true;
            for (const auto& pair : data.metadata) {
                if (!first) oss << ",";
                oss << "\"" << pair.first << "\":\"" << pair.second << "\"";
                first = false;
            }
            oss << "}";
        }
        
        oss << "}";
        return oss.str();
    }
    
    zmq::context_t context_;
    std::unique_ptr<zmq::socket_t> detection_socket_;
    std::unique_ptr<zmq::socket_t> visualization_socket_;
    bool is_initialized_;
};

MessageQueue::MessageQueue() {
    impl_ = std::make_unique<MessageQueueImpl>();
}

MessageQueue::~MessageQueue() {
    Close();
}

bool MessageQueue::Initialize(const std::string& detection_endpoint, 
                              const std::string& visualization_endpoint) {
    return impl_->Initialize(detection_endpoint, visualization_endpoint);
}

bool MessageQueue::SendToDetection(const StructuredData& data) {
    return impl_->SendToDetection(data);
}

bool MessageQueue::SendToVisualization(const StructuredData& data) {
    return impl_->SendToVisualization(data);
}

bool MessageQueue::SendBatch(const std::vector<StructuredData>& data_list) {
    return impl_->SendBatch(data_list);
}

QueueStatus MessageQueue::GetStatus() {
    return impl_->GetStatus();
}

void MessageQueue::Close() {
    impl_->Close();
}

