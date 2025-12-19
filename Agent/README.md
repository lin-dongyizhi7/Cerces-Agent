# Agent 采集端

## 概述

Agent是昇腾NPU分布式训练监测系统的采集端，负责生成模拟的监控指标数据，并通过TCP连接将数据以Protobuf格式流式发送给Communicator。

## 功能特性

- **指标数据生成**：生成各种类型的监控指标数据（T1-T15, D1, F1-F4, B1-B5, G1-G7）
- **流式发送**：支持定时流式发送指标数据
- **自动重连**：连接断开时自动重连
- **批量发送**：支持批量发送多条指标数据，提高效率

## 架构设计

```
MetricGenerator → AgentClient → [TCP/Protobuf] → Communicator
```

### 核心模块

1. **MetricGenerator**：生成各种类型的监控指标数据
2. **AgentClient**：管理与Communicator的连接，负责数据发送

## 编译

```bash
mkdir build
cd build
cmake ..
make
```

## 运行

```bash
./agent [server_host] [server_port] [node_id] [rank_id] [interval_ms]
```

参数说明：
- `server_host`：Communicator服务器地址（默认：localhost）
- `server_port`：Communicator服务器端口（默认：8888）
- `node_id`：节点ID（默认：node_0）
- `rank_id`：Rank ID（默认：rank_0）
- `interval_ms`：发送间隔，毫秒（默认：1000）

示例：
```bash
# 连接到localhost:8888，节点node_0，rank_0，每1秒发送一次
./agent

# 连接到192.168.1.100:8888，节点node_1，rank_1，每500毫秒发送一次
./agent 192.168.1.100 8888 node_1 rank_1 500
```

## 生成的指标类型

Agent可以生成以下类型的指标数据：

### 全过程监控指标（T系列）
- **T1**：功率（Power）
- **T2**：温度（Temperature）
- **T3**：AI Core占用率
- **T4**：AI CPU占用率
- **T5**：Ctrl CPU占用率
- **T6**：内存占用率
- **T7**：内存带宽占用率

### 分阶段监控指标
- **D1**：DataLoader吞吐量
- **F2**：Matmul FLOPS（前向传播）
- **G1**：HCCL AllReduce带宽（梯度同步）

## 数据格式

发送的数据采用Protobuf格式，包含以下字段：
- `node_id`：节点ID
- `rank_id`：Rank ID（在tags中）
- `timestamp`：时间戳（微秒）
- `metric_name`：指标名称
- `value`：指标值
- `unit`：单位
- `tags`：标签（包含rank_id、step_id、metric_type等）

## 使用场景

1. **单机测试**：在一台机器上运行Agent和Communicator进行测试
2. **多机模拟**：在多台机器上运行多个Agent实例，模拟分布式训练场景
3. **性能测试**：调整发送间隔，测试Communicator的接收性能

## 注意事项

- Agent会自动重连，如果Communicator暂时不可用，Agent会持续尝试重连
- 发送间隔建议设置在100ms以上，避免过于频繁的发送导致网络拥塞
- 多个Agent实例可以使用不同的node_id和rank_id来模拟多机多卡场景

