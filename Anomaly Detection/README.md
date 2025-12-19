# 异常检测层 (Anomaly Detection Layer)

异常检测层负责实时接收来自通信转义层的结构化数据，进行异常检测和预警。

## 目录结构

```
Anomaly Detection/
├── common/                          # 通用模块
│   ├── __init__.py
│   ├── data_structures.py          # 数据结构定义
│   └── base_detector.py             # 检测器基类
├── DetectionEngine/                 # 检测引擎
│   ├── __init__.py
│   ├── detection_engine.py          # 检测引擎主类
│   ├── StatisticalDetector/         # 统计学检测器
│   │   ├── __init__.py
│   │   ├── statistical_detector.py
│   │   ├── three_sigma_detector.py  # 3σ规则检测
│   │   ├── static_threshold_detector.py  # R1规则：静态阈值检测
│   │   ├── cusum_detector.py        # R2规则：CUSUM趋势检测
│   │   └── sliding_window_detector.py  # 滑动窗口对比检测
│   ├── ComparisonDetector/         # 对比分析检测器
│   │   ├── __init__.py
│   │   ├── comparison_detector.py
│   │   ├── throughput_comparison_detector.py  # R3规则：吞吐量对比
│   │   ├── flops_comparison_detector.py      # R4规则：FLOPS对比
│   │   ├── rank_communication_detector.py    # R5规则：Rank通信对比
│   │   ├── dp_group_communication_detector.py  # R6规则：DP组通信对比
│   │   └── history_comparison_detector.py   # 历史迭代对比
│   └── MLDetector/                  # 机器学习检测器
│       ├── __init__.py
│       ├── ml_detector.py
│       ├── feature_extractor.py     # 特征提取器
│       ├── isolation_forest_detector.py  # 孤立森林
│       ├── lof_detector.py          # LOF检测器
│       ├── one_class_svm_detector.py    # One-Class SVM
│       └── autoencoder_detector.py   # 自编码器
├── RuleOrchestrator.py              # 规则编排器
├── AlertManager.py                   # 预警管理
├── DataReceiver.py                   # 数据接收模块
├── ConfigManager.py                  # 配置管理
├── AnomalyDetectionController.py     # 主控制器
├── main.py                           # 主程序入口
├── requirements.txt                  # Python依赖
└── README.md                         # 本文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 配置文件

创建配置文件 `config/detection.yaml`：

```yaml
detection_layer:
  data_receiver:
    message_queue:
      type: "zeromq"
      endpoint: "tcp://localhost:5555"
      buffer_size: 10000
  
  detection_engine:
    statistical:
      enabled: true
      window_size: 100
      sigma_threshold: 3.0
      R1:
        enabled: true
        thresholds:
          T1: {upper: 300, lower: 0, consecutive: 3}
          T2: {upper: 85, lower: 0, consecutive: 3}
      R2:
        enabled: true
        cusum:
          delta: 0.2
          h: 5.0
    
    comparison:
      enabled: true
      R3:
        enabled: true
        window_size: 100
        threshold_ratio: 0.2
      R4:
        enabled: true
        baselines:
          F1: 1000.0
          F2: 2000.0
      R5:
        enabled: true
        threshold_ratio: 0.3
      R6:
        enabled: true
        threshold_ratio: 0.2
    
    ml:
      enabled: true
      methods: ["isolation_forest", "lof"]
      isolation_forest:
        n_estimators: 100
        contamination: 0.1
  
  rule_orchestrator:
    enabled_rules: ["R1", "R2", "R3", "R4", "R5", "R6"]
  
  alert:
    dedup_window: 300
    levels:
      critical: ["R1", "R4"]
      warning: ["R2", "R3"]
      info: ["R7", "R10"]
```

### 2. 运行

```bash
python main.py [config_file]
```

## 检测器说明

### StatisticalDetector（统计学检测器）

包含以下检测方法：
- **ThreeSigmaDetector**: 3σ规则检测
- **StaticThresholdDetector**: R1规则，静态阈值检测
- **CUSUMDetector**: R2规则，CUSUM趋势检测
- **SlidingWindowDetector**: 滑动窗口对比检测

### ComparisonDetector（对比分析检测器）

包含以下检测方法：
- **ThroughputComparisonDetector**: R3规则，训练吞吐量对比
- **FLOPSComparisonDetector**: R4规则，FLOPS对比
- **RankCommunicationDetector**: R5规则，Rank通信对比
- **DPGroupCommunicationDetector**: R6规则，DP组通信对比
- **HistoryComparisonDetector**: 历史迭代对比

### MLDetector（机器学习检测器）

包含以下检测方法：
- **IsolationForestDetector**: 孤立森林
- **LOFDetector**: Local Outlier Factor
- **OneClassSVMDetector**: One-Class SVM
- **AutoEncoderDetector**: 自编码器（需要PyTorch）

## 架构说明

1. **DataReceiver**: 从通信转义层接收数据（通过ZeroMQ）
2. **RuleOrchestrator**: 根据指标类型选择适用的检测器
3. **DetectionEngine**: 管理多个检测器，执行检测
4. **AlertManager**: 处理异常结果，生成预警
5. **AnomalyDetectionController**: 主控制器，协调整个流程

## 扩展检测器

要添加新的检测器：

1. 继承 `BaseDetector` 基类
2. 实现 `detect()`, `update_baseline()`, `get_name()` 方法
3. 在对应的聚合检测器（如 `StatisticalDetector`）中注册

## 注意事项

- ML检测器需要足够的训练样本才能开始检测
- 某些ML检测器（如AutoEncoder）需要PyTorch，如果未安装会自动禁用
- 确保通信转义层已启动并监听正确的端口

