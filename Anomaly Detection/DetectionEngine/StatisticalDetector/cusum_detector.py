"""
CUSUM (Cumulative Sum) detector (R2 rule)
"""

from typing import Optional, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
import uuid
from collections import defaultdict


class CUSUMState:
    """CUSUM algorithm state"""
    
    def __init__(self, delta: float = 0.2, h: float = 5.0):
        self.delta = delta  # Temperature parameter (minimum shift to detect)
        self.h = h  # Decision threshold
        self.s_plus = 0.0  # Positive cumulative sum
        self.s_minus = 0.0  # Negative cumulative sum
        self.baseline = None  # Baseline mean
        self.count = 0  # Sample count
    
    def update(self, value: float, baseline: float):
        """Update CUSUM state with new value"""
        if self.baseline is None:
            self.baseline = baseline
        
        # Calculate deviation from baseline
        deviation = value - baseline
        
        # Update cumulative sums
        self.s_plus = max(0, self.s_plus + deviation - self.delta)
        self.s_minus = max(0, self.s_minus - deviation - self.delta)
        
        self.count += 1
    
    def is_anomaly(self) -> bool:
        """Check if anomaly detected"""
        return self.s_plus > self.h or self.s_minus > self.h
    
    def get_anomaly_score(self) -> float:
        """Get normalized anomaly score"""
        max_sum = max(self.s_plus, self.s_minus)
        return min(1.0, max_sum / (self.h * 2))
    
    def reset(self):
        """Reset CUSUM state"""
        self.s_plus = 0.0
        self.s_minus = 0.0


class CUSUMDetector(BaseDetector):
    """CUSUM detector for R2 rule (detecting gradual trend shifts)"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.r2_config = config.get('R2', {})
        self.enabled_r2 = self.r2_config.get('enabled', True)
        self.cusum_config = self.r2_config.get('cusum', {})
        self.delta = self.cusum_config.get('delta', 0.2)  # Minimum shift to detect
        self.h = self.cusum_config.get('h', 5.0)  # Decision threshold
        self.applicable_metrics = self.r2_config.get('applicable_metrics', 
                                                     ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'])
        self.baseline_window_size = self.cusum_config.get('baseline_window_size', 50)
        self.cusum_states: Dict[str, CUSUMState] = {}
        self.baseline_values: Dict[str, list] = defaultdict(list)  # For calculating baseline
    
    def get_name(self) -> str:
        return "CUSUMDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute CUSUM detection (R2 rule)"""
        if not self.enabled or not self.enabled_r2:
            return None
        
        # R2 applies to specified metrics (T1-T7 by default)
        if data.metric_type not in self.applicable_metrics:
            return None
        
        metric_key = f"{data.metric_name}_{data.node_id}_{data.rank_id}"
        
        # Initialize CUSUM state if not exists
        if metric_key not in self.cusum_states:
            self.cusum_states[metric_key] = CUSUMState(self.delta, self.h)
        
        # Update baseline
        self.baseline_values[metric_key].append(data.value)
        if len(self.baseline_values[metric_key]) > self.baseline_window_size:
            self.baseline_values[metric_key].pop(0)
        
        # Calculate baseline mean
        baseline = sum(self.baseline_values[metric_key]) / len(self.baseline_values[metric_key])
        
        # Need enough samples for baseline
        if len(self.baseline_values[metric_key]) < 10:
            return None
        
        # Update CUSUM state
        cusum_state = self.cusum_states[metric_key]
        cusum_state.update(data.value, baseline)
        
        # Check for anomaly
        if cusum_state.is_anomaly():
            anomaly_score = cusum_state.get_anomaly_score()
            
            # Determine trend direction
            trend = 'increasing' if cusum_state.s_plus > self.h else 'decreasing'
            
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=baseline,
                rule_name='R2',
                detector_name=self.get_name(),
                anomaly_score=anomaly_score,
                severity='critical' if anomaly_score > 0.7 else 'warning',
                message=f"R2: {data.metric_name} shows {trend} trend shift "
                       f"(CUSUM+: {cusum_state.s_plus:.2f}, CUSUM-: {cusum_state.s_minus:.2f}, "
                       f"baseline: {baseline:.2f})",
                timestamp_us=data.timestamp_us,
                context={
                    'trend': trend,
                    'cusum_plus': cusum_state.s_plus,
                    'cusum_minus': cusum_state.s_minus,
                    'baseline': baseline,
                    'delta': self.delta,
                    'h': self.h
                }
            )
        
        return None
    
    def update_baseline(self, data: StructuredData):
        """Update baseline"""
        metric_key = f"{data.metric_name}_{data.node_id}_{data.rank_id}"
        self.baseline_values[metric_key].append(data.value)
        if len(self.baseline_values[metric_key]) > self.baseline_window_size:
            self.baseline_values[metric_key].pop(0)

