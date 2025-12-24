# Visualization Layer

可视化层提供实时监控数据推送、检测方法选择接口，并为根因分析预留占位接口。

## 功能

- WebSocket 实时推送：`/ws/realtime`，订阅来自 Communicator 的实时数据（ZeroMQ SUB）。
- 检测方法选择 API：`GET/PUT /api/detection/config`（开启/关闭 statistical、comparison、ml）。
- 根因分析占位：`GET /api/root_cause/{anomaly_id}` 返回占位信息。
- 健康检查：`GET /health`。

## 运行

```bash
cd Visualization
pip install -r requirements.txt
python app.py   # 默认监听 0.0.0.0:5000
```

环境变量：

- `VIS_ENDPOINT`：ZeroMQ 订阅地址，默认 `tcp://localhost:5556`（与 Communicator 中 `visualization_endpoint` 对应）。
- `PORT`：服务端口，默认 5000。

## 与 Communicator 的数据流

- Communicator 已支持 ZeroMQ PUB，将结构化数据发送到 `visualization_endpoint`（默认 5556）。
- Visualization 通过 ZMQ SUB 直接订阅，并推送到 WebSocket 客户端，同时追加到 `logs/realtime.log` 以便后续存储或日志分析。

## 目录结构

```
Visualization/
├── app.py              # FastAPI 服务，ZMQ 订阅 + WebSocket 推送
├── requirements.txt    # 依赖
└── README.md
```
