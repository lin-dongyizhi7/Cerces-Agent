"""
Throughput comparison detector (R3 rule)
"""

from typing import Optional, Dict
from collections import deque, defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
import uuid


class ThroughputComparisonDetector(BaseDetector):
    """Throughput comparison detector for R3 rule"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.r3_config = config.get('R3', {})
        self.enabled_r3 = self.r3_config.get('enabled', True)
        self.window_size = self.r3_config.get('window_size', 100)
        self.threshold_ratio = self.r3_config.get('threshold_ratio', 0.2)  # 20% deviation threshold
        # Store throughput values: metric_key -> deque
        self.throughput_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window_size))
        # Store baseline throughput: metric_key -> baseline_value
        self.baseline_throughput: Dict[str, float] = {}
        self.min_window_size = self.r3_config.get('min_window_size', 20)
    
    def get_name(self) -> str:
        return "ThroughputComparisonDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute throughput comparison detection (R3 rule)"""
        if not self.enabled or not self.enabled_r3:
            return None
        
        # R3 applies to D1 metric (DataLoader throughput)
        if data.metric_type != 'D1':
            return None
        
        metric_key = f"{data.metric_name}_{data.node_id}_{data.rank_id}"
        
        # Update throughput window
        self.throughput_windows[metric_key].append(data.value)
        
        # Need enough samples
        if len(self.throughput_windows[metric_key]) < self.min_window_size:
            # Update baseline if not set
            if metric_key not in self.baseline_throughput:
                self.baseline_throughput[metric_key] = data.value
            return None
        
        # Calculate current average throughput
        current_throughput = sum(self.throughput_windows[metric_key]) / len(self.throughput_windows[metric_key])
        
        # Update baseline if not set
        if metric_key not in self.baseline_throughput:
            self.baseline_throughput[metric_key] = current_throughput
            return None
        
        baseline = self.baseline_throughput[metric_key]
        
        # Check if current throughput deviates significantly from baseline
        if baseline == 0:
            return None
        
        deviation_ratio = abs(current_throughput - baseline) / baseline
        
        if deviation_ratio > self.threshold_ratio:
            # Calculate anomaly score
            anomaly_score = min(1.0, deviation_ratio / (self.threshold_ratio * 2))
            
            trend = 'decreased' if current_throughput < baseline else 'increased'
            
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=baseline * (1 - self.threshold_ratio) if current_throughput < baseline 
                        else baseline * (1 + self.threshold_ratio),
                rule_name='R3',
                detector_name=self.get_name(),
                anomaly_score=anomaly_score,
                severity='critical' if deviation_ratio > 0.5 else 'warning',
                message=f"R3: Training throughput {trend} significantly. "
                       f"Current: {current_throughput:.2f}, Baseline: {baseline:.2f}, "
                       f"Deviation: {deviation_ratio*100:.1f}%",
                timestamp_us=data.timestamp_us,
                context={
                    'current_throughput': current_throughput,
                    'baseline_throughput': baseline,
                    'deviation_ratio': deviation_ratio,
                    'trend': trend,
                    'window_size': len(self.throughput_windows[metric_key])
                }
            )
        
        return None
    
    def update_baseline(self, data: StructuredData):
        """Update baseline throughput"""
        if data.metric_type == 'D1':
            metric_key = f"{data.metric_name}_{data.node_id}_{data.rank_id}"
            if metric_key not in self.baseline_throughput:
                self.baseline_throughput[metric_key] = data.value
            else:
                # Update baseline using exponential moving average
                alpha = 0.1  # Smoothing factor
                self.baseline_throughput[metric_key] = (
                    alpha * data.value + (1 - alpha) * self.baseline_throughput[metric_key]
                )


