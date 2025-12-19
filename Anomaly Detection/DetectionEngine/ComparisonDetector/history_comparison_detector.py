"""
History comparison detector
"""

from typing import Optional, Dict
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
import uuid


class HistoryComparisonDetector(BaseDetector):
    """History iteration comparison detector"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.history_config = config.get('history_comparison', {})
        self.enabled_history = self.history_config.get('enabled', True)
        self.threshold_ratio = self.history_config.get('threshold_ratio', 0.15)  # 15% deviation
        # Store history data: metric_key -> {step_id: value}
        self.history_data: Dict[str, Dict[int, float]] = defaultdict(dict)
        self.max_history_steps = self.history_config.get('max_history_steps', 100)
    
    def get_name(self) -> str:
        return "HistoryComparisonDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute history comparison detection"""
        if not self.enabled or not self.enabled_history:
            return None
        
        metric_key = f"{data.metric_name}_{data.node_id}_{data.rank_id}"
        
        # Store current value
        self.history_data[metric_key][data.step_id] = data.value
        
        # Clean old history
        if len(self.history_data[metric_key]) > self.max_history_steps:
            oldest_step = min(self.history_data[metric_key].keys())
            del self.history_data[metric_key][oldest_step]
        
        # Need at least 2 steps for comparison
        if len(self.history_data[metric_key]) < 2:
            return None
        
        # Get previous step value
        previous_steps = sorted([s for s in self.history_data[metric_key].keys() if s < data.step_id])
        if not previous_steps:
            return None
        
        # Compare with previous step
        previous_step = previous_steps[-1]
        previous_value = self.history_data[metric_key][previous_step]
        
        if previous_value == 0:
            return None
        
        # Calculate change ratio
        change_ratio = abs(data.value - previous_value) / abs(previous_value)
        
        if change_ratio > self.threshold_ratio:
            # Calculate anomaly score
            anomaly_score = min(1.0, change_ratio / (self.threshold_ratio * 2))
            
            trend = 'increased' if data.value > previous_value else 'decreased'
            
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=previous_value,
                rule_name='HistoryComparison',
                detector_name=self.get_name(),
                anomaly_score=anomaly_score,
                severity='warning',
                message=f"History comparison: {data.metric_name} {trend} significantly from step {previous_step} "
                       f"to step {data.step_id}. Previous: {previous_value:.2f}, Current: {data.value:.2f}, "
                       f"Change: {change_ratio*100:.1f}%",
                timestamp_us=data.timestamp_us,
                context={
                    'previous_step': previous_step,
                    'previous_value': previous_value,
                    'current_step': data.step_id,
                    'change_ratio': change_ratio,
                    'trend': trend
                }
            )
        
        return None
    
    def update_baseline(self, data: StructuredData):
        """Update history data"""
        metric_key = f"{data.metric_name}_{data.node_id}_{data.rank_id}"
        self.history_data[metric_key][data.step_id] = data.value

