#include "PythonInterface.h"
#include <iostream>
#include <zmq.hpp>
#include <sstream>
#include <thread>
#include <chrono>

class PythonInterface::PythonInterfaceImpl {
public:
    PythonInterfaceImpl() : context_(1), python_socket_(nullptr), is_available_(false) {}
    
    ~PythonInterfaceImpl() {
        Shutdown();
    }
    
    bool Initialize() {
        try {
            // 使用ZeroMQ PUSH socket连接到Python层
            // Python层使用PULL socket接收数据
            python_socket_ = std::make_unique<zmq::socket_t>(context_, ZMQ_PUSH);
            python_socket_->connect("tcp://localhost:5557"); // Python接口端口
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            is_available_ = true;
            
            std::cout << "PythonInterface initialized" << std::endl;
            return true;
        } catch (const zmq::error_t& e) {
            std::cerr << "Failed to initialize PythonInterface: " << e.what() << std::endl;
            is_available_ = false;
            return false;
        }
    }
    
    void Shutdown() {
        is_available_ = false;
        python_socket_.reset();
    }
    
    bool SendToPythonDetection(const StructuredData& data) {
        if (!is_available_ || !python_socket_) {
            return false;
        }
        
        try {
            // 序列化为JSON格式发送给Python
            std::string json = SerializeToJson(data);
            
            zmq::message_t message(json.size());
            memcpy(message.data(), json.c_str(), json.size());
            
            python_socket_->send(message, zmq::send_flags::dontwait);
            return true;
        } catch (const zmq::error_t& e) {
            std::cerr << "Failed to send to Python: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool SendToPythonDetectionBatch(const std::vector<StructuredData>& data_list) {
        bool all_success = true;
        for (const auto& data : data_list) {
            if (!SendToPythonDetection(data)) {
                all_success = false;
            }
        }
        return all_success;
    }
    
    bool ReceiveFromPython(std::vector<StructuredData>& data_list) {
        // 预留接口：从Python接收数据（如果需要双向通信）
        // 当前实现为空
        (void)data_list;
        return false;
    }
    
    bool IsAvailable() const {
        return is_available_;
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
    std::unique_ptr<zmq::socket_t> python_socket_;
    bool is_available_;
};

PythonInterface::PythonInterface() : is_initialized_(false) {
    impl_ = std::make_unique<PythonInterfaceImpl>();
}

PythonInterface::~PythonInterface() {
    Shutdown();
}

bool PythonInterface::Initialize() {
    if (impl_->Initialize()) {
        is_initialized_ = true;
        return true;
    }
    return false;
}

void PythonInterface::Shutdown() {
    impl_->Shutdown();
    is_initialized_ = false;
}

bool PythonInterface::SendToPythonDetection(const StructuredData& data) {
    return impl_->SendToPythonDetection(data);
}

bool PythonInterface::SendToPythonDetectionBatch(const std::vector<StructuredData>& data_list) {
    return impl_->SendToPythonDetectionBatch(data_list);
}

bool PythonInterface::ReceiveFromPython(std::vector<StructuredData>& data_list) {
    return impl_->ReceiveFromPython(data_list);
}

bool PythonInterface::IsAvailable() const {
    return impl_->IsAvailable();
}

// C接口实现
extern "C" {
    void* CreatePythonInterface() {
        return new PythonInterface();
    }
    
    void DestroyPythonInterface(void* handle) {
        if (handle) {
            delete static_cast<PythonInterface*>(handle);
        }
    }
    
    int SendDataToPython(void* handle, const StructuredData* data) {
        if (!handle || !data) {
            return -1;
        }
        PythonInterface* interface = static_cast<PythonInterface*>(handle);
        return interface->SendToPythonDetection(*data) ? 0 : -1;
    }
}

