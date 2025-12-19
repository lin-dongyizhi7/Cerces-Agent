#include "NetworkManager.h"
#include <iostream>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>

// 网络管理器实现类
class NetworkManagerImpl {
public:
    NetworkManagerImpl() : server_fd_(-1), is_running_(false) {}
    
    ~NetworkManagerImpl() {
        Stop();
    }
    
    bool Start(int port, int thread_count) {
        if (is_running_) {
            return false;
        }
        
        // 创建socket
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            std::cerr << "Failed to create socket: " << strerror(errno) << std::endl;
            return false;
        }
        
        // 设置socket选项
        int opt = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        // 设置为非阻塞模式
        int flags = fcntl(server_fd_, F_GETFL, 0);
        fcntl(server_fd_, F_SETFL, flags | O_NONBLOCK);
        
        // 绑定地址
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);
        
        if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "Failed to bind port " << port << ": " << strerror(errno) << std::endl;
            close(server_fd_);
            server_fd_ = -1;
            return false;
        }
        
        // 开始监听
        if (listen(server_fd_, 100) < 0) {
            std::cerr << "Failed to listen: " << strerror(errno) << std::endl;
            close(server_fd_);
            server_fd_ = -1;
            return false;
        }
        
        is_running_ = true;
        stats_.active_connections = 0;
        stats_.total_bytes_received = 0;
        stats_.total_messages_received = 0;
        stats_.connection_errors = 0;
        
        // 启动工作线程
        for (int i = 0; i < thread_count; ++i) {
            threads_.emplace_back(&NetworkManagerImpl::WorkerThread, this);
        }
        
        // 启动接受连接线程
        accept_thread_ = std::thread(&NetworkManagerImpl::AcceptThread, this);
        
        std::cout << "NetworkManager started on port " << port << " with " 
                  << thread_count << " worker threads" << std::endl;
        return true;
    }
    
    void Stop() {
        if (!is_running_) {
            return;
        }
        
        is_running_ = false;
        
        // 关闭服务器socket
        if (server_fd_ >= 0) {
            close(server_fd_);
            server_fd_ = -1;
        }
        
        // 关闭所有客户端连接
        {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            for (auto& pair : connections_) {
                close(pair.first);
            }
            connections_.clear();
        }
        
        // 等待线程结束
        if (accept_thread_.joinable()) {
            accept_thread_.join();
        }
        
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        std::cout << "NetworkManager stopped" << std::endl;
    }
    
    void RegisterCallback(std::function<void(const char*, size_t)> callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        data_callback_ = callback;
    }
    
    NetworkStats GetStats() {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }
    
    ConnectionStatus GetConnectionStatus(const std::string& client_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        // 简化实现：检查是否有活跃连接
        return connections_.empty() ? ConnectionStatus::DISCONNECTED : ConnectionStatus::CONNECTED;
    }
    
private:
    void AcceptThread() {
        while (is_running_) {
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &addr_len);
            
            if (client_fd >= 0) {
                // 设置为非阻塞
                int flags = fcntl(client_fd, F_GETFL, 0);
                fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);
                
                std::lock_guard<std::mutex> lock(connections_mutex_);
                connections_[client_fd] = std::string(inet_ntoa(client_addr.sin_addr));
                stats_.active_connections = connections_.size();
                
                std::cout << "New connection from " << inet_ntoa(client_addr.sin_addr) 
                          << ":" << ntohs(client_addr.sin_port) << std::endl;
            } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.connection_errors++;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    void WorkerThread() {
        const size_t buffer_size = 65536;
        char buffer[buffer_size];
        
        while (is_running_) {
            std::vector<int> to_remove;
            
            {
                std::lock_guard<std::mutex> lock(connections_mutex_);
                for (auto& pair : connections_) {
                    int client_fd = pair.first;
                    
                    ssize_t n = recv(client_fd, buffer, buffer_size, 0);
                    
                    if (n > 0) {
                        // 收到数据
                        {
                            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                            stats_.total_bytes_received += n;
                            stats_.total_messages_received++;
                        }
                        
                        // 调用回调
                        std::lock_guard<std::mutex> callback_lock(callback_mutex_);
                        if (data_callback_) {
                            data_callback_(buffer, n);
                        }
                    } else if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
                        // 连接关闭或错误
                        to_remove.push_back(client_fd);
                    }
                }
                
                // 移除关闭的连接
                for (int fd : to_remove) {
                    close(fd);
                    connections_.erase(fd);
                    stats_.active_connections = connections_.size();
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    int server_fd_;
    std::atomic<bool> is_running_;
    std::map<int, std::string> connections_;
    std::mutex connections_mutex_;
    
    std::function<void(const char*, size_t)> data_callback_;
    std::mutex callback_mutex_;
    
    NetworkStats stats_;
    std::mutex stats_mutex_;
    
    std::thread accept_thread_;
    std::vector<std::thread> threads_;
};

// NetworkManager实现
NetworkManager::NetworkManager() : is_running_(false) {
    impl_ = std::make_unique<NetworkManagerImpl>();
}

NetworkManager::~NetworkManager() {
    StopServer();
}

bool NetworkManager::StartServer(int port, int thread_count) {
    if (impl_->Start(port, thread_count)) {
        is_running_ = true;
        return true;
    }
    return false;
}

void NetworkManager::StopServer() {
    impl_->Stop();
    is_running_ = false;
}

void NetworkManager::RegisterDataCallback(std::function<void(const char*, size_t)> callback) {
    impl_->RegisterCallback(callback);
}

ConnectionStatus NetworkManager::GetConnectionStatus(const std::string& client_id) {
    return impl_->GetConnectionStatus(client_id);
}

NetworkStats NetworkManager::GetStatistics() {
    return impl_->GetStats();
}

bool NetworkManager::IsRunning() const {
    return is_running_;
}

