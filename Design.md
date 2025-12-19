# 昇腾NPU分布式训练监测系统Center端设计文档

## 1. 系统架构设计

### 1.1 总体架构

Center端采用分层架构设计，各层之间通过消息队列或RPC进行通信，实现松耦合和高内聚。

```
┌─────────────────────────────────────────────────────────┐
│                    可视化层 (Visualization Layer)        │
│              Python + Web (Flask/FastAPI + HTML/CSS/JS) │
└────────────────────┬────────────────────────────────────┘
                     │ WebSocket / HTTP
┌────────────────────┴────────────────────────────────────┐
│              根因分析层 (Root Cause Analysis Layer)      │
│              Python (因果图谱 + 图神经网络)             │
└────────────────────┬────────────────────────────────────┘
                     │ 异常事件触发
┌────────────────────┴────────────────────────────────────┐
│              异常检测层 (Anomaly Detection Layer)        │
│              Python (多种异常检测算法)                   │
└────────────────────┬────────────────────────────────────┘
                     │ 结构化数据流
┌────────────────────┴────────────────────────────────────┐
│              通信转义层 (Communication Layer)            │
│              C++ (Protobuf接收与转义)                   │
└────────────────────┬────────────────────────────────────┘
                     │ TCP/UDP
┌────────────────────┴────────────────────────────────────┐
│                   采集端 (Agent)                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 技术选型

#### 1.2.1 通信转义层
- **语言**：C++
- **原因**：需要高性能的数据接收和处理，C++在性能上有明显优势
- **关键库**：
  - Protobuf：协议解析
  - Boost.Asio或libevent：异步网络IO
  - ZeroMQ或RabbitMQ：消息队列

#### 1.2.2 异常检测层
- **语言**：Python
- **原因**：
  - 异常检测算法丰富，Python生态完善（scikit-learn、PyOD等）
  - 便于快速迭代和算法实验
  - 与根因分析层和可视化层技术栈统一
- **关键库**：
  - scikit-learn：经典机器学习算法
  - PyOD：异常检测专用库
  - NumPy/Pandas：数据处理
  - Redis/RabbitMQ：消息队列

#### 1.2.3 根因分析层
- **语言**：Python
- **原因**：图神经网络和因果分析库主要基于Python
- **关键库**：
  - PyTorch Geometric或DGL：图神经网络
  - NetworkX：图分析
  - pgmpy：概率图模型
  - DoWhy：因果推断

#### 1.2.4 可视化层
- **后端**：Python (Flask或FastAPI)
- **前端**：原生HTML/CSS/JavaScript + Chart.js/ECharts
- **原因**：
  - 原生技术栈性能更好，减少框架开销
  - Chart.js/ECharts提供高性能图表渲染
  - WebSocket实现实时数据推送
- **备选方案**：React/Vue（如需要更复杂的交互）

### 1.3 数据流设计

```
采集端 → [Protobuf] → 通信转义层 → [结构化数据] → 异常检测层
                                                      ↓
                                                 [异常事件]
                                                      ↓
                                                根因分析层
                                                      ↓
                                                 [分析结果]
                                                      ↓
采集端 → [Protobuf] → 通信转义层 → [实时数据流] → 可视化层 ← [异常报警] ← 异常检测层
                                                      ↑
                                                 [根因分析结果] ← 根因分析层
```

## 2. 通信转义层详细设计

### 2.1 模块划分

```
通信转义层
├── NetworkManager        # 网络连接管理
├── ProtocolHandler       # Protobuf协议处理
├── DataTransformer       # 数据转义模块
├── MessageQueue          # 消息队列管理
└── ConfigManager         # 配置管理
```

### 2.2 核心类设计

#### 2.2.1 NetworkManager
**职责**：管理网络连接，接收来自多机的数据流

**主要接口**：
```cpp
class NetworkManager {
public:
    // 启动服务器，监听指定端口
    bool StartServer(int port, int thread_count);
    
    // 停止服务器
    void StopServer();
    
    // 注册数据接收回调
    void RegisterDataCallback(std::function<void(const char*, size_t)> callback);
    
    // 获取连接状态
    ConnectionStatus GetConnectionStatus(const std::string& client_id);
    
    // 获取统计信息
    NetworkStats GetStatistics();
};
```

**设计要点**：
- 使用异步IO模型（epoll/kqueue），支持高并发
- 维护连接池，管理多个客户端连接
- 实现心跳机制，检测连接健康状态
- 支持断线重连

#### 2.2.2 ProtocolHandler
**职责**：解析Protobuf协议数据

**主要接口**：
```cpp
class ProtocolHandler {
public:
    // 解析Protobuf消息
    bool ParseMessage(const char* data, size_t length, MonitorData& output);
    
    // 验证消息完整性
    bool ValidateMessage(const MonitorData& data);
    
    // 获取协议版本
    int GetProtocolVersion(const char* data, size_t length);
};
```

**Protobuf消息定义**（示例）：
```protobuf
syntax = "proto3";

message MonitorData {
    string node_id = 1;           // 节点ID
    int64 timestamp = 2;          // 时间戳（微秒）
    string metric_name = 3;       // 指标名称
    double value = 4;             // 指标值
    string unit = 5;              // 单位
    map<string, string> tags = 6; // 标签（rank_id, step_id等）
}

message BatchMonitorData {
    repeated MonitorData data = 1;
}
```

#### 2.2.3 DataTransformer
**职责**：将Protobuf数据转换为系统内部结构化数据

**主要接口**：
```cpp
class DataTransformer {
public:
    // 转换单条数据
    StructuredData Transform(const MonitorData& proto_data);
    
    // 批量转换
    std::vector<StructuredData> TransformBatch(
        const BatchMonitorData& batch_data);
    
    // 数据标准化
    void Normalize(StructuredData& data);
};
```

**结构化数据格式**：
```cpp
struct StructuredData {
    std::string node_id;
    std::string rank_id;
    int64_t timestamp_us;      // 微秒时间戳
    std::string metric_type;   // 指标类型（T1-T15, D1, F1-F4, B1-B5, G1-G7）
    std::string metric_name;   // 指标名称
    double value;              // 指标值
    std::string unit;          // 单位
    int32_t step_id;           // 训练步数
    std::map<std::string, std::string> metadata; // 元数据
};
```

#### 2.2.4 MessageQueue
**职责**：管理消息队列，向下一层发送数据

**主要接口**：
```cpp
class MessageQueue {
public:
    // 发送数据到异常检测层
    bool SendToDetection(const StructuredData& data);
    
    // 发送数据到可视化层
    bool SendToVisualization(const StructuredData& data);
    
    // 批量发送
    bool SendBatch(const std::vector<StructuredData>& data_list);
    
    // 获取队列状态
    QueueStatus GetStatus();
};
```

**实现方案**：
- 使用ZeroMQ或RabbitMQ作为消息中间件
- 异常检测层使用PUSH/PULL模式
- 可视化层使用PUB/SUB模式（支持多订阅者）

### 2.3 性能优化设计

1. **内存池**：使用内存池减少内存分配开销
2. **批量处理**：支持批量接收和转义，提高吞吐量
3. **异步处理**：使用异步IO，避免阻塞
4. **数据压缩**：对大数据量进行压缩传输

### 2.4 配置设计

```yaml
communication_layer:
  server:
    port: 8888
    thread_count: 8
    max_connections: 100
    heartbeat_interval: 30  # 秒
  
  protobuf:
    max_message_size: 10485760  # 10MB
    version: "v1.0"
  
  message_queue:
    type: "zeromq"  # 或 "rabbitmq"
    detection_endpoint: "tcp://localhost:5555"
    visualization_endpoint: "tcp://localhost:5556"
    buffer_size: 10000
```

## 3. 异常检测层详细设计

### 3.1 模块划分

```
异常检测层
├── DataReceiver          # 数据接收模块
├── DetectionEngine       # 检测引擎
│   ├── StatisticalDetector    # 统计学检测器
│   ├── ComparisonDetector     # 对比分析检测器
│   └── MLDetector             # 机器学习检测器
├── RuleEngine            # 规则引擎（实现R1-R10规则）
├── AlertManager          # 预警管理
└── ConfigManager         # 配置管理
```

### 3.2 核心类设计

#### 3.2.1 DetectionEngine
**职责**：异常检测引擎，管理多种检测方法

**主要接口**：
```python
class DetectionEngine:
    def __init__(self, config: Dict):
        """初始化检测引擎，加载配置的检测方法"""
        pass
    
    def detect(self, data: StructuredData) -> List[AnomalyResult]:
        """执行异常检测，返回检测结果列表"""
        pass
    
    def register_detector(self, detector: BaseDetector):
        """注册检测器"""
        pass
    
    def get_detection_results(self, 
                             data: StructuredData) -> Dict[str, AnomalyResult]:
        """获取所有检测方法的结果"""
        pass
```

#### 3.2.2 BaseDetector（抽象基类）
**职责**：定义检测器接口规范

**接口定义**：
```python
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """执行检测，返回异常结果或None"""
        pass
    
    @abstractmethod
    def update_baseline(self, data: StructuredData):
        """更新基线数据"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取检测器名称"""
        pass
```

#### 3.2.3 StatisticalDetector（统计学检测器）
**实现方法**：
- 3σ规则检测
- 静态阈值检测（R1）
- CUSUM趋势检测（R2）
- 滑动窗口对比检测

**类设计**：
```python
class StatisticalDetector(BaseDetector):
    def __init__(self, config: Dict):
        self.window_size = config.get('window_size', 100)
        self.sigma_threshold = config.get('sigma_threshold', 3.0)
        self.static_thresholds = config.get('static_thresholds', {})
        self.cusum_config = config.get('cusum', {})
        self.sliding_windows = {}  # metric_name -> SlidingWindow
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """执行统计学检测"""
        # 1. 更新滑动窗口
        # 2. 执行3σ检测
        # 3. 执行静态阈值检测
        # 4. 执行CUSUM检测
        # 5. 返回异常结果
        pass
```

#### 3.2.4 ComparisonDetector（对比分析检测器）
**实现方法**：
- Rank对比检测
- 历史迭代对比检测
- DP组内/跨组对比检测

**类设计**：
```python
class ComparisonDetector(BaseDetector):
    def __init__(self, config: Dict):
        self.rank_comparison_enabled = config.get('rank_comparison', True)
        self.history_comparison_enabled = config.get('history_comparison', True)
        self.dp_group_comparison_enabled = config.get('dp_group_comparison', True)
        self.history_data = {}  # step_id -> metrics
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """执行对比分析检测"""
        # 1. Rank对比
        # 2. 历史迭代对比
        # 3. DP组对比
        pass
```

#### 3.2.5 MLDetector（机器学习检测器）
**实现方法**：
- 孤立森林
- LOF
- One-Class SVM
- 自编码器

**类设计**：
```python
class MLDetector(BaseDetector):
    def __init__(self, config: Dict):
        self.method = config.get('method', 'isolation_forest')
        self.model = self._load_model()
        self.feature_extractor = FeatureExtractor()
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """执行机器学习检测"""
        features = self.feature_extractor.extract(data)
        anomaly_score = self.model.predict([features])[0]
        if anomaly_score < threshold:
            return AnomalyResult(...)
        return None
```

#### 3.2.6 RuleEngine（规则引擎）
**职责**：实现R1-R10检测规则

**类设计**：
```python
class RuleEngine:
    def __init__(self, config: Dict):
        self.rules = {
            'R1': StaticThresholdRule(config['R1']),
            'R2': CUSUMRule(config['R2']),
            'R3': ThroughputComparisonRule(config['R3']),
            'R4': FLOPSComparisonRule(config['R4']),
            'R5': RankCommunicationRule(config['R5']),
            'R6': DPGroupCommunicationRule(config['R6']),
            'R7': MinorKernelRule(config['R7']),
            'R8': KernelLaunchDelayRule(config['R8']),
            'R9': MemoryCopyRateRule(config['R9']),
            'R10': InterStepCPURule(config['R10']),
        }
    
    def execute_rules(self, data: StructuredData) -> List[AnomalyResult]:
        """执行所有规则"""
        results = []
        for rule_name, rule in self.rules.items():
            if rule.is_applicable(data):
                result = rule.check(data)
                if result:
                    results.append(result)
        return results
```

#### 3.2.7 AlertManager（预警管理）
**职责**：管理异常预警

**类设计**：
```python
class AlertManager:
    def __init__(self, config: Dict):
        self.alert_levels = {
            'critical': 3,
            'warning': 2,
            'info': 1
        }
        self.deduplication_window = config.get('dedup_window', 300)  # 秒
        self.recent_alerts = {}  # 用于去重
    
    def process_anomaly(self, anomaly: AnomalyResult):
        """处理异常，生成预警"""
        # 1. 确定预警等级
        # 2. 去重检查
        # 3. 生成预警事件
        # 4. 发送到可视化层和根因分析层
        pass
    
    def send_alert(self, alert: Alert):
        """发送预警"""
        # 发送到消息队列或WebSocket
        pass
```

### 3.3 数据存储设计

**滑动窗口存储**：
- 使用Redis存储滑动窗口数据（TTL=1小时）
- 数据结构：Sorted Set（按时间戳排序）

**基线数据存储**：
- 使用Redis存储动态基线（均值、标准差等）
- 定期持久化到数据库

**历史数据存储**：
- 使用时序数据库（InfluxDB或TimescaleDB）存储历史数据
- 支持快速查询和聚合

### 3.4 配置设计

```yaml
detection_layer:
  detectors:
    statistical:
      enabled: true
      window_size: 100
      sigma_threshold: 3.0
      cusum:
        delta: 0.2  # 温度
        h: 5
    
    comparison:
      enabled: true
      rank_comparison: true
      history_comparison: true
      dp_group_comparison: true
    
    ml:
      enabled: true
      methods: ["isolation_forest", "lof"]
      isolation_forest:
        n_estimators: 100
        contamination: 0.1
  
  rules:
    R1:
      enabled: true
      thresholds:
        T1: {upper: 300, lower: 0, consecutive: 3}
        T2: {upper: 85, lower: 0, consecutive: 3}
        # ...
    R2:
      enabled: true
      # ...
  
  alert:
    dedup_window: 300
    levels:
      critical: ["R1", "R4"]
      warning: ["R2", "R3"]
      info: ["R7", "R10"]
```

## 4. 根因分析层详细设计

### 4.1 模块划分

```
根因分析层
├── CausalGraphAnalyzer   # 因果图谱分析器
├── GNNAnalyzer           # 图神经网络分析器
├── CorrelationAnalyzer  # 关联分析器
└── ResultAggregator      # 结果聚合器
```

### 4.2 核心类设计

#### 4.2.1 CausalGraphAnalyzer（因果图谱分析器）
**职责**：基于因果图谱进行根因分析

**类设计**：
```python
class CausalGraphAnalyzer:
    def __init__(self, config: Dict):
        self.causal_graph = None
        self.graph_builder = CausalGraphBuilder()
        self.inference_engine = CausalInferenceEngine()
    
    def build_graph(self, historical_data: List[StructuredData]):
        """基于历史数据构建因果图谱"""
        # 1. 使用PC算法或LiNGAM算法发现因果关系
        # 2. 构建有向无环图（DAG）
        # 3. 计算因果强度
        self.causal_graph = self.graph_builder.build(historical_data)
    
    def analyze(self, anomaly: AnomalyResult, 
                context_data: Dict) -> RootCauseResult:
        """执行根因分析"""
        # 1. 在因果图谱中找到异常节点
        # 2. 反向追溯根因节点
        # 3. 计算根因可能性
        # 4. 返回根因分析结果
        pass
```

**因果图谱构建方法**：
- PC算法：基于条件独立性测试
- LiNGAM：线性非高斯无环模型
- 基于时间序列的Granger因果检验

#### 4.2.2 GNNAnalyzer（图神经网络分析器）
**职责**：使用图神经网络进行根因分析

**类设计**：
```python
class GNNAnalyzer:
    def __init__(self, config: Dict):
        self.model = self._build_gnn_model()
        self.graph_constructor = SystemGraphConstructor()
    
    def _build_gnn_model(self) -> torch.nn.Module:
        """构建GNN模型"""
        # 使用GraphSAGE或GCN
        return GraphSAGE(
            in_channels=feature_dim,
            hidden_channels=128,
            out_channels=1,
            num_layers=3
        )
    
    def construct_system_graph(self, 
                              metrics: Dict) -> dgl.DGLGraph:
        """构建系统图"""
        # 节点：指标/组件
        # 边：指标间关系（相关性、因果关系等）
        return self.graph_constructor.construct(metrics)
    
    def analyze(self, anomaly: AnomalyResult,
                system_graph: dgl.DGLGraph) -> RootCauseResult:
        """使用GNN进行根因分析"""
        # 1. 提取节点特征
        # 2. 使用GNN进行异常传播分析
        # 3. 识别根因节点
        pass
```

#### 4.2.3 RootCauseResult（根因分析结果）
**数据结构**：
```python
@dataclass
class RootCauseResult:
    anomaly_id: str
    root_causes: List[RootCause]  # 按可能性排序
    confidence_scores: Dict[str, float]  # 根因置信度
    propagation_paths: List[List[str]]  # 异常传播路径
    suggestions: List[str]  # 优化建议
    analysis_time: float  # 分析耗时

@dataclass
class RootCause:
    metric_name: str
    node_id: str
    rank_id: str
    probability: float
    impact_scope: List[str]  # 影响范围
    evidence: Dict  # 证据信息
```

### 4.3 图神经网络模型设计

**模型架构**：
- **输入**：系统图（节点特征、边特征）
- **编码器**：GraphSAGE或GCN，学习节点表示
- **异常检测头**：MLP，输出异常分数
- **根因定位头**：Attention机制，定位根因节点

**训练数据**：
- 使用历史异常数据进行监督学习
- 或使用无监督方法（自编码器）

### 4.4 配置设计

```yaml
root_cause_analysis:
  causal_graph:
    enabled: true
    algorithm: "pc"  # 或 "lingam"
    update_interval: 3600  # 秒
    min_samples: 1000
  
  gnn:
    enabled: true
    model_type: "graphsage"  # 或 "gcn"
    hidden_dim: 128
    num_layers: 3
    learning_rate: 0.001
  
  correlation:
    enabled: true
    method: "pearson"  # 或 "spearman"
    threshold: 0.7
```

## 5. 可视化层详细设计

### 5.1 架构设计

```
可视化层
├── Backend (Flask/FastAPI)
│   ├── API Server        # RESTful API
│   ├── WebSocket Server  # 实时数据推送
│   └── Data Service      # 数据服务
└── Frontend
    ├── Real-time Monitor  # 实时监控页面
    ├── Alert Dashboard   # 报警仪表盘
    └── Root Cause View   # 根因分析页面
```

### 5.2 后端设计

#### 5.2.1 API Server
**主要接口**：

```python
# Flask/FastAPI路由设计
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """获取指标列表"""
    pass

@app.route('/api/metrics/<metric_name>/data', methods=['GET'])
def get_metric_data(metric_name, start_time, end_time):
    """获取指标历史数据"""
    pass

@app.route('/api/alerts', methods=['GET'])
def get_alerts(start_time, end_time, level=None):
    """获取报警列表"""
    pass

@app.route('/api/alerts/<alert_id>', methods=['GET'])
def get_alert_detail(alert_id):
    """获取报警详情"""
    pass

@app.route('/api/root_cause/<anomaly_id>', methods=['GET'])
def get_root_cause(anomaly_id):
    """获取根因分析结果"""
    pass

@app.route('/api/root_cause/<anomaly_id>/analyze', methods=['POST'])
def trigger_root_cause_analysis(anomaly_id):
    """触发根因分析"""
    pass
```

#### 5.2.2 WebSocket Server
**实时数据推送**：

```python
@app.websocket('/ws/realtime')
async def websocket_realtime(websocket):
    """实时数据WebSocket连接"""
    # 1. 接收客户端订阅的指标列表
    # 2. 从消息队列订阅数据
    # 3. 实时推送数据到客户端
    pass

@app.websocket('/ws/alerts')
async def websocket_alerts(websocket):
    """报警推送WebSocket连接"""
    # 实时推送报警信息
    pass
```

#### 5.2.3 Data Service
**数据服务类**：

```python
class DataService:
    def __init__(self):
        self.redis_client = redis.Redis()
        self.db_client = InfluxDBClient()
        self.message_queue = MessageQueue()
    
    def get_realtime_data(self, metric_names: List[str]) -> Dict:
        """获取实时数据"""
        pass
    
    def get_historical_data(self, metric_name: str, 
                           start_time: int, end_time: int) -> List[Dict]:
        """获取历史数据"""
        pass
    
    def subscribe_realtime(self, metric_names: List[str], 
                          callback: Callable):
        """订阅实时数据"""
        pass
```

### 5.3 前端设计

#### 5.3.1 实时监控页面
**功能模块**：
- 指标选择器：多选指标
- 时间范围选择器
- 动态图表：使用Chart.js或ECharts
- 节点选择器：选择要显示的节点/rank

**技术实现**：
```javascript
// 实时数据更新
const ws = new WebSocket('ws://localhost:5000/ws/realtime');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateChart(data);
};

// 图表初始化
const chart = new Chart(ctx, {
    type: 'line',
    data: {
        datasets: []
    },
    options: {
        animation: false,
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'second'
                }
            }
        }
    }
});
```

#### 5.3.2 报警仪表盘
**功能模块**：
- 报警列表：实时显示报警
- 报警统计：今日报警数、各等级分布
- 报警详情：点击查看详情
- 报警过滤：按时间、等级、指标过滤

**界面设计**：
```html
<div class="alert-dashboard">
    <div class="alert-stats">
        <div class="stat-item">
            <span class="label">今日报警</span>
            <span class="value" id="today-alerts">0</span>
        </div>
        <!-- 更多统计 -->
    </div>
    <div class="alert-list">
        <table id="alert-table">
            <!-- 报警列表 -->
        </table>
    </div>
</div>
```

#### 5.3.3 根因分析页面
**功能模块**：
- 异常选择：选择要分析的异常
- 分析触发：触发根因分析
- Loading状态：分析进行中显示
- 结果展示：根因列表、因果图谱可视化、传播路径

**界面设计**：
```html
<div class="root-cause-view">
    <div class="anomaly-selector">
        <select id="anomaly-select">
            <!-- 异常列表 -->
        </select>
        <button id="analyze-btn">开始分析</button>
    </div>
    <div id="loading" class="loading" style="display: none;">
        <div class="spinner"></div>
        <p>正在分析中...</p>
    </div>
    <div id="result" class="result" style="display: none;">
        <div class="root-causes">
            <h3>根因列表</h3>
            <ul id="root-cause-list"></ul>
        </div>
        <div class="causal-graph">
            <h3>因果图谱</h3>
            <div id="graph-container"></div>
        </div>
    </div>
</div>
```

**因果图谱可视化**：
- 使用D3.js或Cytoscape.js绘制交互式图谱
- 节点表示指标/组件，边表示因果关系
- 支持缩放、拖拽、高亮等交互

### 5.4 性能优化

1. **数据采样**：前端对大数据量进行采样显示
2. **虚拟滚动**：报警列表使用虚拟滚动
3. **图表优化**：使用Canvas渲染，限制数据点数量
4. **缓存策略**：缓存历史数据查询结果

### 5.5 配置设计

```yaml
visualization:
  backend:
    host: "0.0.0.0"
    port: 5000
    workers: 4
  
  frontend:
    chart_library: "echarts"  # 或 "chartjs"
    update_interval: 1000  # 毫秒
    max_data_points: 1000
  
  websocket:
    heartbeat_interval: 30
    max_connections: 100
```

## 6. 数据存储设计

### 6.1 存储架构

```
数据存储
├── Redis          # 实时数据缓存、滑动窗口
├── InfluxDB       # 时序数据存储（历史数据）
└── PostgreSQL     # 关系数据存储（报警、根因分析结果）
```

### 6.2 数据模型设计

#### 6.2.1 Redis数据结构
- **实时数据**：`realtime:{metric_name}:{node_id}` → Sorted Set (timestamp, value)
- **滑动窗口**：`window:{metric_name}:{node_id}` → Sorted Set
- **基线数据**：`baseline:{metric_name}:{node_id}` → Hash (mean, std, ...)

#### 6.2.2 InfluxDB数据模型
**Measurement**: `monitor_metrics`
**Tags**: node_id, rank_id, metric_type, metric_name
**Fields**: value, unit
**Time**: timestamp

#### 6.2.3 PostgreSQL数据模型
**alerts表**：
```sql
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    anomaly_id VARCHAR(64) UNIQUE,
    metric_name VARCHAR(64),
    node_id VARCHAR(64),
    rank_id VARCHAR(64),
    alert_level VARCHAR(16),
    value DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    rule_name VARCHAR(16),
    timestamp BIGINT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**root_cause_analysis表**：
```sql
CREATE TABLE root_cause_analysis (
    id SERIAL PRIMARY KEY,
    anomaly_id VARCHAR(64),
    root_causes JSONB,
    confidence_scores JSONB,
    propagation_paths JSONB,
    suggestions TEXT[],
    analysis_time DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 7. 部署设计

### 7.1 容器化部署

**Docker Compose配置**：

```yaml
version: '3.8'

services:
  communication-layer:
    build: ./communication_layer
    ports:
      - "8888:8888"
    environment:
      - CONFIG_PATH=/config/communication.yaml
  
  detection-layer:
    build: ./detection_layer
    depends_on:
      - redis
      - influxdb
    environment:
      - CONFIG_PATH=/config/detection.yaml
  
  root-cause-layer:
    build: ./root_cause_layer
    depends_on:
      - postgres
    environment:
      - CONFIG_PATH=/config/root_cause.yaml
  
  visualization-layer:
    build: ./visualization_layer
    ports:
      - "5000:5000"
    depends_on:
      - redis
      - influxdb
      - postgres
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
  
  influxdb:
    image: influxdb:2.7
    volumes:
      - influxdb_data:/var/lib/influxdb2
  
  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=monitor
      - POSTGRES_USER=monitor
      - POSTGRES_PASSWORD=password

volumes:
  redis_data:
  influxdb_data:
  postgres_data:
```

### 7.2 分布式部署

各层可独立部署，通过消息队列和网络通信：

```
Node 1: 通信转义层
Node 2: 异常检测层
Node 3: 根因分析层
Node 4: 可视化层
Node 5: Redis + InfluxDB + PostgreSQL
```

## 8. 监控与运维

### 8.1 系统监控
- 各层运行状态监控
- 性能指标监控（CPU、内存、延迟、吞吐量）
- 错误日志监控

### 8.2 日志设计
- 使用结构化日志（JSON格式）
- 日志级别：DEBUG、INFO、WARNING、ERROR
- 日志轮转和归档

### 8.3 告警机制
- 系统自身异常告警
- 资源使用告警
- 服务可用性告警

## 9. 扩展性设计

### 9.1 插件化架构
- 异常检测方法支持插件化扩展
- 根因分析方法支持插件化扩展
- 可视化组件支持插件化扩展

### 9.2 配置化设计
- 所有参数支持配置文件管理
- 支持热更新配置（部分参数）

## 10. 安全设计

### 10.1 数据安全
- 传输加密：TLS/SSL
- 数据脱敏：敏感数据脱敏处理

### 10.2 访问控制
- 用户认证：JWT Token
- 权限管理：RBAC（基于角色的访问控制）

### 10.3 审计日志
- 记录关键操作日志
- 支持审计查询

