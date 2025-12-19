"""
Sliding window comparison detector
"""

from typing import Optional, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
from .three_sigma_detector import SlidingWindow
import uuid


class SlidingWindowDetector(BaseDetector):
    """Sliding window comparison detector"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.window_size = config.get('window_size', 100)
        self.k_sigma = config.get('k_sigma', 2.0)  # k * sigma threshold
        self.sliding_windows: Dict[str, SlidingWindow] = {}
        self.min_window_size = config.get('min_window_size', 20)
    
    def get_name(self) -> str:
        return "SlidingWindowDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute sliding window comparison detection"""
        if not self.enabled:
            return None
        
        metric_key = f"{data.metric_name}_{data.node_id}_{data.rank_id}"
        
        # Update sliding window
        if metric_key not in self.sliding_windows:
            self.sliding_windows[metric_key] = SlidingWindow(self.window_size)
        
        window = self.sliding_windows[metric_key]
        window.add(data.value, data.timestamp_us)
        
        # Need enough samples
        if window.get_count() < self.min_window_size:
            return None
        
        # Calculate mean and std
        mean = window.get_mean()
        std = window.get_std()
        
        if std == 0:
            return None
        
        # Check if value exceeds mean ± k*sigma
        upper_bound = mean + self.k_sigma * std
        lower_bound = mean - self.k_sigma * std
        
        is_anomaly = data.value > upper_bound or data.value < lower_bound
        
        if is_anomaly:
            # Calculate anomaly score
            if data.value > upper_bound:
                deviation = (data.value - upper_bound) / std if std > 0 else 0
            else:
                deviation = (lower_bound - data.value) / std if std > 0 else 0
            
            anomaly_score = min(1.0, deviation / (self.k_sigma * 2))
            
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=upper_bound if data.value > upper_bound else lower_bound,
                rule_name='SlidingWindow',
                detector_name=self.get_name(),
                anomaly_score=anomaly_score,
                severity='warning',
                message=f"Value {data.value:.2f} exceeds sliding window bounds "
                       f"({lower_bound:.2f}, {upper_bound:.2f}), "
                       f"mean={mean:.2f}, std={std:.2f}",
                timestamp_us=data.timestamp_us,
                context={
                    'mean': mean,
                    'std': std,
                    'upper_bound': upper_bound,
                    'lower_bound': lower_bound,
                    'k_sigma': self.k_sigma,
                    'window_size': window.get_count()
                }
            )
        
        return None
    
    def update_baseline(self, data: StructuredData):
        """Update baseline (sliding window)"""
        metric_key = f"{data.metric_name}_{data.node_id}_{data.rank_id}"
        if metric_key not in self.sliding_windows:
            self.sliding_windows[metric_key] = SlidingWindow(self.window_size)
        self.sliding_windows[metric_key].add(data.value, data.timestamp_us)

