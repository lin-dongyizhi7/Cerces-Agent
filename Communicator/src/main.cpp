#include "NetworkManager.h"
#include "ProtocolHandler.h"
#include "DataTransformer.h"
#include "MessageQueue.h"
#include "ConfigManager.h"
#include "PythonInterface.h"
#include "monitor.pb.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <atomic>
#include <fstream>
#include <filesystem>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#ifdef __linux__
#include <unistd.h>
#endif

std::atomic<bool> g_running(true);

void SignalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

// Helper function to find config file in multiple possible locations
std::string FindConfigFile(const std::string& default_path) {
    // If path is absolute or explicitly provided, try it first
    if (std::filesystem::path(default_path).is_absolute() || 
        std::filesystem::exists(default_path)) {
        if (std::filesystem::exists(default_path)) {
            return default_path;
        }
    }
    
    // Try relative to executable location (for build directory)
    // Get executable path
    std::string exe_path;
    #ifdef __APPLE__
        char path[1024];
        uint32_t size = sizeof(path);
        if (_NSGetExecutablePath(path, &size) == 0) {
            exe_path = std::filesystem::canonical(path).parent_path().string();
        }
    #elif __linux__
        char path[1024];
        ssize_t count = readlink("/proc/self/exe", path, sizeof(path) - 1);
        if (count != -1) {
            path[count] = '\0';
            exe_path = std::filesystem::canonical(path).parent_path().string();
        }
    #endif
    
    if (!exe_path.empty()) {
        // Try ../config/communicator.conf (from build directory)
        std::filesystem::path config_path = std::filesystem::path(exe_path) / ".." / "config" / "communicator.conf";
        if (std::filesystem::exists(config_path)) {
            return std::filesystem::canonical(config_path).string();
        }
        // Try ../../config/communicator.conf (if executable is in build/subdirectory)
        config_path = std::filesystem::path(exe_path) / ".." / ".." / "config" / "communicator.conf";
        if (std::filesystem::exists(config_path)) {
            return std::filesystem::canonical(config_path).string();
        }
    }
    
    // Try relative to current working directory
    if (std::filesystem::exists("config/communicator.conf")) {
        return std::filesystem::canonical("config/communicator.conf").string();
    }
    
    // Try in parent directory
    if (std::filesystem::exists("../config/communicator.conf")) {
        return std::filesystem::canonical("../config/communicator.conf").string();
    }
    
    // Return original path if nothing found (will fail gracefully)
    return default_path;
}

int main(int argc, char* argv[]) {
    // 注册信号处理
    signal(SIGINT, SignalHandler);
    signal(SIGTERM, SignalHandler);
    
    // 加载配置
    std::string config_path = (argc > 1) ? argv[1] : "config/communicator.conf";
    // Try to find config file in multiple locations
    config_path = FindConfigFile(config_path);
    ConfigManager& config = ConfigManager::GetInstance();
    if (!config.LoadFromFile(config_path)) {
        std::cerr << "Warning: Failed to load config file, using defaults" << std::endl;
    }
    
    // 读取配置
    int port = config.GetInt("server.port", 8888);
    int thread_count = config.GetInt("server.thread_count", 4);
    std::string detection_endpoint = config.GetString("message_queue.detection_endpoint", 
                                                       "tcp://localhost:5555");
    std::string visualization_endpoint = config.GetString("message_queue.visualization_endpoint",
                                                           "tcp://localhost:5556");
    bool enable_python_interface = config.GetBool("python_interface.enabled", true);
    
    // 初始化组件
    NetworkManager network_manager;
    ProtocolHandler protocol_handler;
    DataTransformer data_transformer;
    MessageQueue message_queue;
    PythonInterface python_interface;
    
    // 初始化消息队列
    if (!message_queue.Initialize(detection_endpoint, visualization_endpoint)) {
        std::cerr << "Failed to initialize MessageQueue" << std::endl;
        return 1;
    }
    
    // 初始化Python接口（如果启用）
    if (enable_python_interface) {
        if (!python_interface.Initialize()) {
            std::cerr << "Warning: Failed to initialize PythonInterface" << std::endl;
        }
    }
    
    // 注册数据接收回调
    network_manager.RegisterDataCallback([&](const char* data, size_t length) {
        // 检查消息是否完整
        size_t message_length = 0;
        if (!protocol_handler.IsMessageComplete(data, length, message_length)) {
            // 消息不完整，需要缓冲（简化实现中直接返回）
            return;
        }
        
        // 尝试解析批量消息
        monitor::BatchMonitorData batch_data;
        if (protocol_handler.ParseBatchMessage(data, message_length, batch_data)) {
            // 批量转换数据
            auto structured_data_list = data_transformer.TransformBatch(batch_data);
            
            // 发送到消息队列
            message_queue.SendBatch(structured_data_list);
            
            // 发送到Python接口（如果启用）
            if (enable_python_interface && python_interface.IsAvailable()) {
                python_interface.SendToPythonDetectionBatch(structured_data_list);
            }
        } else {
            // 尝试解析单条消息
            monitor::MonitorData proto_data;
            if (protocol_handler.ParseMessage(data, message_length, proto_data)) {
                // 转换数据
                auto structured_data = data_transformer.Transform(proto_data);
                
                // 发送到消息队列
                message_queue.SendToDetection(structured_data);
                message_queue.SendToVisualization(structured_data);
                
                // 发送到Python接口（如果启用）
                if (enable_python_interface && python_interface.IsAvailable()) {
                    python_interface.SendToPythonDetection(structured_data);
                }
            }
        }
    });
    
    // 启动服务器
    if (!network_manager.StartServer(port, thread_count)) {
        std::cerr << "Failed to start server on port " << port << std::endl;
        return 1;
    }
    
    std::cout << "Communicator started successfully on port " << port << std::endl;
    std::cout << "Press Ctrl+C to stop..." << std::endl;
    
    // 主循环：定期打印统计信息
    while (g_running && network_manager.IsRunning()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        
        auto stats = network_manager.GetStatistics();
        std::cout << "Stats - Connections: " << stats.active_connections
                  << ", Messages: " << stats.total_messages_received
                  << ", Bytes: " << stats.total_bytes_received << std::endl;
    }
    
    // 清理
    network_manager.StopServer();
    message_queue.Close();
    if (enable_python_interface) {
        python_interface.Shutdown();
    }
    
    std::cout << "Communicator stopped" << std::endl;
    return 0;
}

