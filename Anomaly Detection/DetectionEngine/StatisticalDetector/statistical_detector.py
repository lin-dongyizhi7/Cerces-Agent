"""
Statistical detector - aggregates multiple statistical and comparison-based detection methods
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
from .throughput_comparison_detector import ThroughputComparisonDetector
from .flops_comparison_detector import FLOPSComparisonDetector
from .rank_communication_detector import RankCommunicationDetector
from .dp_group_communication_detector import DPGroupCommunicationDetector
from .history_comparison_detector import HistoryComparisonDetector


class StatisticalDetector(BaseDetector):
    """
    Unified statistical detector.

    This detector aggregates:
    - single-metric statistical methods (e.g., 3-sigma, static threshold, CUSUM, sliding window)
    - comparison-based statistical methods (e.g., throughput/FLOPS/rank/DP-group/history comparison)
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.detectors: List[BaseDetector] = []

        # ------------------------------
        # Single-metric statistical methods
        # ------------------------------
        statistical_config = config.get('statistical', {})

        # 3-sigma detector
        if statistical_config.get('three_sigma', {}).get('enabled', True):
            three_sigma_config = dict(statistical_config.get('three_sigma', {}))
            three_sigma_config.update(config)  # Merge parent config
            self.detectors.append(ThreeSigmaDetector(three_sigma_config))

        # R1: Static threshold detector
        if config.get('R1', {}).get('enabled', True):
            r1_config = dict(config.get('R1', {}))
            r1_config.update(config)  # Merge parent config
            self.detectors.append(StaticThresholdDetector(r1_config))

        # R2: CUSUM detector
        if config.get('R2', {}).get('enabled', True):
            r2_config = dict(config.get('R2', {}))
            r2_config.update(config)  # Merge parent config
            self.detectors.append(CUSUMDetector(r2_config))

        # Sliding window detector
        if statistical_config.get('sliding_window', {}).get('enabled', True):
            sliding_window_config = dict(statistical_config.get('sliding_window', {}))
            sliding_window_config.update(config)  # Merge parent config
            self.detectors.append(SlidingWindowDetector(sliding_window_config))

        # ------------------------------
        # Comparison-based statistical methods (R3-R6, history comparison)
        # ------------------------------
        comparison_config = config.get('comparison', {})

        # R3: Throughput comparison detector
        if config.get('R3', {}).get('enabled', True):
            r3_config = dict(config.get('R3', {}))
            r3_config.update(config)
            self.detectors.append(ThroughputComparisonDetector(r3_config))

        # R4: FLOPS comparison detector
        if config.get('R4', {}).get('enabled', True):
            r4_config = dict(config.get('R4', {}))
            r4_config.update(config)
            self.detectors.append(FLOPSComparisonDetector(r4_config))

        # R5: Rank communication detector
        if config.get('R5', {}).get('enabled', True):
            r5_config = dict(config.get('R5', {}))
            r5_config.update(config)
            self.detectors.append(RankCommunicationDetector(r5_config))

        # R6: DP group communication detector
        if config.get('R6', {}).get('enabled', True):
            r6_config = dict(config.get('R6', {}))
            r6_config.update(config)
            self.detectors.append(DPGroupCommunicationDetector(r6_config))

        # History comparison detector (iteration-level comparison)
        if comparison_config.get('history_comparison', {}).get('enabled', True):
            history_config = dict(comparison_config.get('history_comparison', {}))
            history_config.update(config)
            self.detectors.append(HistoryComparisonDetector(history_config))
    
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

