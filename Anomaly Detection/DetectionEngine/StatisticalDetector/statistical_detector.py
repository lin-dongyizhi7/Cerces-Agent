"""
Statistical detector - aggregates multiple statistical detection methods
"""

from typing import Optional, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
from .three_sigma_detector import ThreeSigmaDetector
from .static_threshold_detector import StaticThresholdDetector
from .cusum_detector import CUSUMDetector
from .sliding_window_detector import SlidingWindowDetector


class StatisticalDetector(BaseDetector):
    """Statistical detector that aggregates multiple statistical detection methods"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.detectors: List[BaseDetector] = []
        
        # Initialize sub-detectors
        statistical_config = config.get('statistical', {})
        
        # 3-sigma detector
        if statistical_config.get('three_sigma', {}).get('enabled', True):
            three_sigma_config = statistical_config.get('three_sigma', {})
            three_sigma_config.update(config)  # Merge parent config
            self.detectors.append(ThreeSigmaDetector(three_sigma_config))
        
        # R1: Static threshold detector
        if config.get('R1', {}).get('enabled', True):
            r1_config = config.get('R1', {})
            r1_config.update(config)  # Merge parent config
            self.detectors.append(StaticThresholdDetector(r1_config))
        
        # R2: CUSUM detector
        if config.get('R2', {}).get('enabled', True):
            r2_config = config.get('R2', {})
            r2_config.update(config)  # Merge parent config
            self.detectors.append(CUSUMDetector(r2_config))
        
        # Sliding window detector
        if statistical_config.get('sliding_window', {}).get('enabled', True):
            sliding_window_config = statistical_config.get('sliding_window', {})
            sliding_window_config.update(config)  # Merge parent config
            self.detectors.append(SlidingWindowDetector(sliding_window_config))
    
    def get_name(self) -> str:
        return "StatisticalDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute statistical detection using all sub-detectors"""
        if not self.enabled:
            return None
        
        # Try each detector, return first anomaly found
        # In practice, you might want to aggregate results from multiple detectors
        for detector in self.detectors:
            if detector.is_enabled():
                result = detector.detect(data)
                if result:
                    return result
        
        return None
    
    def update_baseline(self, data: StructuredData):
        """Update baseline for all sub-detectors"""
        for detector in self.detectors:
            if detector.is_enabled():
                detector.update_baseline(data)

