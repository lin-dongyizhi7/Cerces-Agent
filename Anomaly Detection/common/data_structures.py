"""
Data structures for anomaly detection
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime


@dataclass
class StructuredData:
    """Structured data format from communication layer"""
    node_id: str
    rank_id: str
    timestamp_us: int  # Microsecond timestamp
    metric_type: str  # Metric type (T1-T15, D1, F1-F4, B1-B5, G1-G7)
    metric_name: str  # Metric name
    value: float  # Metric value
    unit: str  # Unit
    step_id: int  # Training step ID
    metadata: Dict[str, str] = field(default_factory=dict)  # Metadata
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StructuredData':
        """Create StructuredData from dictionary"""
        return cls(
            node_id=data.get('node_id', ''),
            rank_id=data.get('rank_id', ''),
            timestamp_us=data.get('timestamp_us', 0),
            metric_type=data.get('metric_type', ''),
            metric_name=data.get('metric_name', ''),
            value=data.get('value', 0.0),
            unit=data.get('unit', ''),
            step_id=data.get('step_id', 0),
            metadata=data.get('metadata', {})
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'node_id': self.node_id,
            'rank_id': self.rank_id,
            'timestamp_us': self.timestamp_us,
            'metric_type': self.metric_type,
            'metric_name': self.metric_name,
            'value': self.value,
            'unit': self.unit,
            'step_id': self.step_id,
            'metadata': self.metadata
        }


@dataclass
class AnomalyResult:
    """Anomaly detection result"""
    anomaly_id: str  # Unique anomaly ID
    metric_name: str  # Metric name
    metric_type: str  # Metric type
    node_id: str  # Node ID
    rank_id: str  # Rank ID
    value: float  # Anomalous value
    threshold: Optional[float] = None  # Threshold value
    rule_name: str = ''  # Rule name (R1-R10)
    detector_name: str = ''  # Detector name
    anomaly_score: float = 0.0  # Anomaly score
    severity: str = 'warning'  # Severity: critical, warning, info
    message: str = ''  # Description message
    timestamp_us: int = 0  # Timestamp when anomaly detected
    context: Dict = field(default_factory=dict)  # Additional context
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'anomaly_id': self.anomaly_id,
            'metric_name': self.metric_name,
            'metric_type': self.metric_type,
            'node_id': self.node_id,
            'rank_id': self.rank_id,
            'value': self.value,
            'threshold': self.threshold,
            'rule_name': self.rule_name,
            'detector_name': self.detector_name,
            'anomaly_score': self.anomaly_score,
            'severity': self.severity,
            'message': self.message,
            'timestamp_us': self.timestamp_us,
            'context': self.context
        }


@dataclass
class Alert:
    """Alert information"""
    alert_id: str  # Unique alert ID
    anomaly_result: AnomalyResult  # Related anomaly result
    alert_level: str  # Alert level: critical, warning, info
    created_at: datetime  # Alert creation time
    acknowledged: bool = False  # Whether alert is acknowledged
    resolved: bool = False  # Whether alert is resolved
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'anomaly_result': self.anomaly_result.to_dict(),
            'alert_level': self.alert_level,
            'created_at': self.created_at.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }

