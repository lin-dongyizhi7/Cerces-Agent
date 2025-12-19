# Communicator 通信转义层

## 概述

Communicator是昇腾NPU分布式训练监测系统的通信转义层，负责接收来自Agent的Protobuf格式监控数据，进行协议解析和数据转义，然后将结构化数据发送到异常检测层和可视化层。

## 功能特性

- **高性能网络接收**：使用异步IO模型，支持多机并发接入
- **Protobuf协议解析**：解析和验证来自Agent的Protobuf消息
- **数据转义**：将Protobuf数据转换为系统内部结构化数据格式
- **消息队列分发**：通过ZeroMQ将数据发送到异常检测层和可视化层
- **Python接口**：预留与Python层进行数据对接的接口

## 架构设计

```
Agent → [TCP] → NetworkManager → ProtocolHandler → DataTransformer → MessageQueue → 异常检测层/可视化层
                                                                    ↓
                                                              PythonInterface → Python层
```

### 核心模块

1. **NetworkManager**：管理网络连接，接收来自Agent的数据流
2. **ProtocolHandler**：解析Protobuf协议数据
3. **DataTransformer**：将Protobuf数据转换为结构化数据
4. **MessageQueue**：管理消息队列，向下一层发送数据
5. **ConfigManager**：配置管理
6. **PythonInterface**：预留的Python接口

## 编译

```bash
mkdir build
cd build
cmake ..
make
```

## 运行

```bash
./communicator [config_file]
```

默认配置文件路径：`config/communicator.conf`

## 配置说明

配置文件采用key=value格式，主要配置项：

- `server.port`：监听端口（默认8888）
- `server.thread_count`：工作线程数（默认4）
- `message_queue.detection_endpoint`：异常检测层消息队列端点
- `message_queue.visualization_endpoint`：可视化层消息队列端点
- `python_interface.enabled`：是否启用Python接口

## Python接口

Communicator预留了与Python层进行数据对接的接口，通过ZeroMQ PUSH socket发送数据到Python层（端口5557）。Python层可以使用PULL socket接收数据。

### C接口

提供了C风格的接口，方便Python通过ctypes调用：

```c
void* CreatePythonInterface();
void DestroyPythonInterface(void* handle);
int SendDataToPython(void* handle, const StructuredData* data);
```

## 数据格式

### 输入格式（Protobuf）

```protobuf
message MonitorData {
    string node_id = 1;
    int64 timestamp = 2;
    string metric_name = 3;
    double value = 4;
    string unit = 5;
    map<string, string> tags = 6;
}
```

### 输出格式（结构化数据）

```cpp
struct StructuredData {
    std::string node_id;
    std::string rank_id;
    int64_t timestamp_us;
    std::string metric_type;
    std::string metric_name;
    double value;
    std::string unit;
    int32_t step_id;
    std::map<std::string, std::string> metadata;
};
```

## 性能指标

- 支持单机接收吞吐量≥100MB/s
- 数据接收到转义完成延迟≤10ms（P99）
- 支持≥100个并发连接
- CPU占用率≤30%，内存占用≤2GB

