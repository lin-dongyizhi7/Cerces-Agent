"""
Rule orchestrator - manages rule to detector mapping
"""

from typing import List, Dict
import sys
import os

sys.path.append(os.path.dirname(__file__))
from common.data_structures import StructuredData


class RuleOrchestrator:
    """Rule orchestrator that manages rule to detector mapping"""
    
    def __init__(self, config: dict):
        """
        Initialize rule orchestrator
        
        Args:
            config: Configuration dictionary
        """
        # Rule to detector mapping
        self.rule_to_detector_map = {
            # R1, R2 rules implemented by StatisticalDetector
            'R1': 'statistical',
            'R2': 'statistical',
            # R3-R6 rules implemented by ComparisonDetector
            'R3': 'comparison',
            'R4': 'comparison',
            'R5': 'comparison',
            'R6': 'comparison',
            # R7-R10 rules (can be implemented by StatisticalDetector or other detectors)
            'R7': 'statistical',  # Minor NPU kernel idle ratio
            'R8': 'statistical',  # Kernel launch delay distribution
            'R9': 'statistical',  # Memory copy rate
            'R10': 'statistical',  # Inter-step CPU operation time
        }
        
        self.enabled_rules = config.get('enabled_rules', 
                                       ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10'])
        self.metric_type_to_rules = self._build_metric_rule_mapping()
    
    def _build_metric_rule_mapping(self) -> Dict[str, List[str]]:
        """
        Build mapping from metric type to applicable rules
        
        Returns:
            Dictionary mapping metric types to rule lists
        """
        return {
            # T1-T7: R1 (static threshold), R2 (CUSUM)
            'T1': ['R1', 'R2'],  # Power
            'T2': ['R1', 'R2'],  # Temperature
            'T3': ['R1', 'R2'],  # AI Core utilization
            'T4': ['R1', 'R2'],  # AI CPU utilization
            'T5': ['R1', 'R2'],  # Ctrl CPU utilization
            'T6': ['R1', 'R2'],  # Memory utilization
            'T7': ['R1', 'R2'],  # Memory bandwidth utilization
            # D1: R3 (throughput comparison)
            'D1': ['R3'],  # DataLoader throughput
            # F1-F4: R4 (FLOPS comparison)
            'F1': ['R4'],  # aclnnFlashAttentionScore
            'F2': ['R4'],  # aclnnMatmul
            'F3': ['R4'],  # aclnnBatchMatMul
            'F4': ['R4'],  # aclnnFFN
            # G1-G4: R5 (rank communication), R6 (DP group communication)
            'G1': ['R5', 'R6'],  # hcclAllReduce
            'G2': ['R5', 'R6'],  # hcclBroadcast
            'G3': ['R5', 'R6'],  # hcclAllGather
            'G4': ['R5', 'R6'],  # hcclReduceScatter
            # R7-R10 can be applied to various metrics based on specific requirements
        }
    
    def get_applicable_detectors(self, data: StructuredData) -> List[str]:
        """
        Get applicable detectors for given data
        
        Args:
            data: Structured data
            
        Returns:
            List of detector names
        """
        applicable_detectors = set()
        metric_type = data.metric_type
        
        # Get applicable rules for this metric type
        applicable_rules = self.metric_type_to_rules.get(metric_type, [])
        
        # Filter enabled rules
        enabled_applicable_rules = [r for r in applicable_rules if r in self.enabled_rules]
        
        # Map rules to detectors
        for rule in enabled_applicable_rules:
            detector = self.rule_to_detector_map.get(rule)
            if detector:
                applicable_detectors.add(detector)
        
        return list(applicable_detectors)
    
    def get_rules_for_detector(self, detector_name: str) -> List[str]:
        """
        Get rules handled by specified detector
        
        Args:
            detector_name: Detector name
            
        Returns:
            List of rule names
        """
        return [rule for rule, detector in self.rule_to_detector_map.items() 
                if detector == detector_name]
    
    def is_rule_enabled(self, rule_name: str) -> bool:
        """
        Check if rule is enabled
        
        Args:
            rule_name: Rule name (R1-R10)
            
        Returns:
            True if enabled, False otherwise
        """
        return rule_name in self.enabled_rules

