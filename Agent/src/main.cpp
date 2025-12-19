#include "AgentClient.h"
#include <iostream>
#include <signal.h>
#include <atomic>
#include <thread>
#include <chrono>

std::atomic<bool> g_running(true);

void SignalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

int main(int argc, char* argv[]) {
    // 注册信号处理
    signal(SIGINT, SignalHandler);
    signal(SIGTERM, SignalHandler);
    
    // 解析命令行参数
    std::string server_host = "localhost";
    int server_port = 8888;
    std::string node_id = "node_0";
    std::string rank_id = "rank_0";
    int interval_ms = 1000; // 发送间隔（毫秒）
    
    if (argc > 1) {
        server_host = argv[1];
    }
    if (argc > 2) {
        server_port = std::stoi(argv[2]);
    }
    if (argc > 3) {
        node_id = argv[3];
    }
    if (argc > 4) {
        rank_id = argv[4];
    }
    if (argc > 5) {
        interval_ms = std::stoi(argv[5]);
    }
    
    std::cout << "Agent starting..." << std::endl;
    std::cout << "  Server: " << server_host << ":" << server_port << std::endl;
    std::cout << "  Node ID: " << node_id << std::endl;
    std::cout << "  Rank ID: " << rank_id << std::endl;
    std::cout << "  Send Interval: " << interval_ms << "ms" << std::endl;
    
    // 创建Agent客户端
    AgentClient client(server_host, server_port, node_id, rank_id);
    
    // 启动客户端
    if (!client.Start()) {
        std::cerr << "Failed to start agent client" << std::endl;
        return 1;
    }
    
    // 启动流式发送
    client.StartStreaming(interval_ms);
    
    std::cout << "Agent started, streaming metrics..." << std::endl;
    std::cout << "Press Ctrl+C to stop..." << std::endl;
    
    // 主循环：等待停止信号
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // 检查连接状态
        if (!client.IsConnected() && g_running) {
            std::cout << "Connection lost, attempting to reconnect..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            client.Start();
        }
    }
    
    // 清理
    client.Stop();
    
    std::cout << "Agent stopped" << std::endl;
    return 0;
}

