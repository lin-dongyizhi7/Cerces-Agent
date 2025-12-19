"""
3-sigma rule detector
"""

import numpy as np
from typing import Optional, Dict
from collections import deque
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
import uuid


class SlidingWindow:
    """Sliding window for storing time series data"""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
    
    def add(self, value: float, timestamp: int):
        """Add value to window"""
        self.values.append(value)
        self.timestamps.append(timestamp)
    
    def get_mean(self) -> float:
        """Get mean value"""
        if len(self.values) == 0:
            return 0.0
        return float(np.mean(self.values))
    
    def get_std(self) -> float:
        """Get standard deviation"""
        if len(self.values) < 2:
            return 0.0
        return float(np.std(self.values, ddof=1))
    
    def get_count(self) -> int:
        """Get current window size"""
        return len(self.values)
    
    def is_full(self) -> bool:
        """Check if window is full"""
        return len(self.values) >= self.window_size


class ThreeSigmaDetector(BaseDetector):
    """3-sigma rule detector"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.window_size = config.get('window_size', 100)
        self.sigma_threshold = config.get('sigma_threshold', 3.0)
        self.sliding_windows: Dict[str, SlidingWindow] = {}
        self.min_window_size = config.get('min_window_size', 10)  # Minimum samples for detection
    
    def get_name(self) -> str:
        return "ThreeSigmaDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute 3-sigma detection"""
        if not self.enabled:
            return None
        
        metric_key = f"{data.metric_name}_{data.node_id}_{data.rank_id}"
        
        # Update sliding window
        if metric_key not in self.sliding_windows:
            self.sliding_windows[metric_key] = SlidingWindow(self.window_size)
        
        window = self.sliding_windows[metric_key]
        window.add(data.value, data.timestamp_us)
        
        # Need enough samples for detection
        if window.get_count() < self.min_window_size:
            return None
        
        # Calculate mean and std
        mean = window.get_mean()
        std = window.get_std()
        
        if std == 0:
            return None
        
        # Check if value exceeds 3-sigma threshold
        z_score = abs(data.value - mean) / std
        
        if z_score > self.sigma_threshold:
            # Calculate anomaly score (normalized to 0-1)
            anomaly_score = min(1.0, z_score / (self.sigma_threshold * 2))
            
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=mean + self.sigma_threshold * std if data.value > mean else mean - self.sigma_threshold * std,
                rule_name='3Sigma',
                detector_name=self.get_name(),
                anomaly_score=anomaly_score,
                severity='warning' if z_score < self.sigma_threshold * 1.5 else 'critical',
                message=f"Value {data.value:.2f} exceeds {self.sigma_threshold}σ threshold "
                       f"(mean={mean:.2f}, std={std:.2f}, z-score={z_score:.2f})",
                timestamp_us=data.timestamp_us,
                context={
                    'mean': mean,
                    'std': std,
                    'z_score': z_score,
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

