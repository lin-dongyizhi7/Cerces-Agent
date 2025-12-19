"""
Static threshold detector (R1 rule)
"""

from typing import Optional, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
import uuid
from collections import defaultdict


class StaticThresholdDetector(BaseDetector):
    """Static threshold detector for R1 rule"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.r1_config = config.get('R1', {})
        self.enabled_r1 = self.r1_config.get('enabled', True)
        self.static_thresholds = self.r1_config.get('thresholds', {})
        self.consecutive_count = self.r1_config.get('consecutive', 3)  # Consecutive violations required
        # Track consecutive violations: metric_key -> count
        self.violation_counts: Dict[str, int] = defaultdict(int)
        # Track last violation timestamp: metric_key -> timestamp
        self.last_violation_timestamp: Dict[str, int] = {}
        self.violation_timeout = self.r1_config.get('violation_timeout', 60)  # seconds
    
    def get_name(self) -> str:
        return "StaticThresholdDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute static threshold detection (R1 rule)"""
        if not self.enabled or not self.enabled_r1:
            return None
        
        # R1 applies to T1-T7 metrics
        if data.metric_type not in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']:
            return None
        
        metric_key = f"{data.metric_name}_{data.node_id}_{data.rank_id}"
        
        # Get threshold for this metric
        threshold_config = self.static_thresholds.get(data.metric_name)
        if not threshold_config:
            return None
        
        upper_threshold = threshold_config.get('upper')
        lower_threshold = threshold_config.get('lower')
        required_consecutive = threshold_config.get('consecutive', self.consecutive_count)
        
        # Check if value exceeds threshold
        is_violation = False
        threshold_type = None
        threshold_value = None
        
        if upper_threshold is not None and data.value > upper_threshold:
            is_violation = True
            threshold_type = 'upper'
            threshold_value = upper_threshold
        elif lower_threshold is not None and data.value < lower_threshold:
            is_violation = True
            threshold_type = 'lower'
            threshold_value = lower_threshold
        
        if is_violation:
            # Check if this is a continuation of previous violation
            last_timestamp = self.last_violation_timestamp.get(metric_key, 0)
            time_diff = (data.timestamp_us - last_timestamp) / 1e6  # Convert to seconds
            
            # Reset count if too much time has passed
            if time_diff > self.violation_timeout:
                self.violation_counts[metric_key] = 1
            else:
                self.violation_counts[metric_key] += 1
            
            self.last_violation_timestamp[metric_key] = data.timestamp_us
            
            # Check if consecutive violations reached threshold
            if self.violation_counts[metric_key] >= required_consecutive:
                # Calculate anomaly score
                excess = abs(data.value - threshold_value) / threshold_value if threshold_value != 0 else 0
                anomaly_score = min(1.0, excess)
                
                severity = 'critical' if excess > 0.2 else 'warning'
                
                return AnomalyResult(
                    anomaly_id=str(uuid.uuid4()),
                    metric_name=data.metric_name,
                    metric_type=data.metric_type,
                    node_id=data.node_id,
                    rank_id=data.rank_id,
                    value=data.value,
                    threshold=threshold_value,
                    rule_name='R1',
                    detector_name=self.get_name(),
                    anomaly_score=anomaly_score,
                    severity=severity,
                    message=f"R1: {data.metric_name} value {data.value:.2f} exceeds {threshold_type} threshold "
                           f"{threshold_value:.2f} for {self.violation_counts[metric_key]} consecutive times",
                    timestamp_us=data.timestamp_us,
                    context={
                        'threshold_type': threshold_type,
                        'consecutive_count': self.violation_counts[metric_key],
                        'required_consecutive': required_consecutive
                    }
                )
        else:
            # Reset violation count if value is within threshold
            self.violation_counts[metric_key] = 0
        
        return None
    
    def update_baseline(self, data: StructuredData):
        """Update baseline (thresholds are static, no update needed)"""
        pass

