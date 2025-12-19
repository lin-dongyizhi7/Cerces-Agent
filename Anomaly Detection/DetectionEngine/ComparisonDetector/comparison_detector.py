"""
Comparison detector - aggregates multiple comparison detection methods
"""

from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
from .throughput_comparison_detector import ThroughputComparisonDetector
from .flops_comparison_detector import FLOPSComparisonDetector
from .rank_communication_detector import RankCommunicationDetector
from .dp_group_communication_detector import DPGroupCommunicationDetector
from .history_comparison_detector import HistoryComparisonDetector


class ComparisonDetector(BaseDetector):
    """Comparison detector that aggregates multiple comparison detection methods"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.detectors = []
        
        comparison_config = config.get('comparison', {})
        
        # R3: Throughput comparison detector
        if config.get('R3', {}).get('enabled', True):
            r3_config = config.get('R3', {})
            r3_config.update(config)
            self.detectors.append(ThroughputComparisonDetector(r3_config))
        
        # R4: FLOPS comparison detector
        if config.get('R4', {}).get('enabled', True):
            r4_config = config.get('R4', {})
            r4_config.update(config)
            self.detectors.append(FLOPSComparisonDetector(r4_config))
        
        # R5: Rank communication detector
        if config.get('R5', {}).get('enabled', True):
            r5_config = config.get('R5', {})
            r5_config.update(config)
            self.detectors.append(RankCommunicationDetector(r5_config))
        
        # R6: DP group communication detector
        if config.get('R6', {}).get('enabled', True):
            r6_config = config.get('R6', {})
            r6_config.update(config)
            self.detectors.append(DPGroupCommunicationDetector(r6_config))
        
        # History comparison detector
        if comparison_config.get('history_comparison', {}).get('enabled', True):
            history_config = comparison_config.get('history_comparison', {})
            history_config.update(config)
            self.detectors.append(HistoryComparisonDetector(history_config))
    
    def get_name(self) -> str:
        return "ComparisonDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute comparison detection using all sub-detectors"""
        if not self.enabled:
            return None
        
        # Try each detector, return first anomaly found
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

