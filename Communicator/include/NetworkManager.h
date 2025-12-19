#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H

#include <functional>
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>

// 连接状态
enum class ConnectionStatus {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    ERROR
};

// 网络统计信息
struct NetworkStats {
    uint64_t total_bytes_received;
    uint64_t total_messages_received;
    uint64_t active_connections;
    uint64_t connection_errors;
};

// 前向声明
class NetworkManagerImpl;

class NetworkManager {
public:
    NetworkManager();
    ~NetworkManager();

    // 启动服务器，监听指定端口
    bool StartServer(int port, int thread_count = 4);
    
    // 停止服务器
    void StopServer();
    
    // 注册数据接收回调
    void RegisterDataCallback(std::function<void(const char*, size_t)> callback);
    
    // 获取连接状态
    ConnectionStatus GetConnectionStatus(const std::string& client_id);
    
    // 获取统计信息
    NetworkStats GetStatistics();
    
    // 检查服务器是否运行
    bool IsRunning() const;

private:
    std::unique_ptr<NetworkManagerImpl> impl_;
    std::atomic<bool> is_running_;
};

#endif // NETWORK_MANAGER_H

