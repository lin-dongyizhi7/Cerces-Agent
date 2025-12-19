"""
DP group communication detector (R6 rule)
"""

from typing import Optional, Dict, List
from collections import defaultdict
import sys
import os
import statistics

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
import uuid


class DPGroupCommunicationDetector(BaseDetector):
    """DP group communication detector for R6 rule (cross-group comparison)"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.r6_config = config.get('R6', {})
        self.enabled_r6 = self.r6_config.get('enabled', True)
        self.threshold_ratio = self.r6_config.get('threshold_ratio', 0.2)  # 20% deviation threshold
        # Store DP group averages: timestamp -> {dp_group_id: average_value}
        self.dp_group_averages: Dict[int, Dict[str, float]] = defaultdict(dict)
        # Store DP group values: timestamp -> {dp_group_id: [values]}
        self.dp_group_values: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.cache_timeout = 5  # seconds
        self.cache_timeout_us = self.cache_timeout * 1_000_000
    
    def get_name(self) -> str:
        return "DPGroupCommunicationDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute DP group communication detection (R6 rule)"""
        if not self.enabled or not self.enabled_r6:
            return None
        
        # R6 applies to G1-G4 metrics (communication functions)
        if data.metric_type not in ['G1', 'G2', 'G3', 'G4']:
            return None
        
        # Extract DP group from metadata
        dp_group_id = data.metadata.get('dp_group_id', 'default')
        
        # Store value for this DP group
        timestamp_key = data.timestamp_us // 1_000_000  # Round to second
        self.dp_group_values[timestamp_key][dp_group_id].append(data.value)
        
        # Calculate average for this DP group
        group_values = self.dp_group_values[timestamp_key][dp_group_id]
        group_average = statistics.mean(group_values) if group_values else 0
        self.dp_group_averages[timestamp_key][dp_group_id] = group_average
        
        # Clean old cache entries
        current_timestamp_key = data.timestamp_us // 1_000_000
        keys_to_remove = [k for k in self.dp_group_averages.keys() 
                         if current_timestamp_key - k > self.cache_timeout]
        for k in keys_to_remove:
            if k in self.dp_group_averages:
                del self.dp_group_averages[k]
            if k in self.dp_group_values:
                del self.dp_group_values[k]
        
        # Get all DP group averages at this timestamp
        all_group_averages = list(self.dp_group_averages[timestamp_key].values())
        
        if len(all_group_averages) < 2:
            return None
        
        # Calculate overall mean and std across all DP groups
        overall_mean = statistics.mean(all_group_averages)
        overall_std = statistics.stdev(all_group_averages) if len(all_group_averages) > 1 else 0
        
        if overall_mean == 0:
            return None
        
        # Check if this DP group's average deviates significantly from overall mean
        deviation_ratio = abs(group_average - overall_mean) / overall_mean if overall_mean > 0 else 0
        
        if deviation_ratio > self.threshold_ratio:
            # Calculate anomaly score
            anomaly_score = min(1.0, deviation_ratio / (self.threshold_ratio * 2))
            
            z_score = (group_average - overall_mean) / overall_std if overall_std > 0 else 0
            
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=overall_mean * (1 - self.threshold_ratio) if group_average < overall_mean 
                        else overall_mean * (1 + self.threshold_ratio),
                rule_name='R6',
                detector_name=self.get_name(),
                anomaly_score=anomaly_score,
                severity='critical' if deviation_ratio > 0.4 else 'warning',
                message=f"R6: DP group {dp_group_id} communication bandwidth deviates from other groups. "
                       f"Group average: {group_average:.2f}, Overall mean: {overall_mean:.2f}, "
                       f"Deviation: {deviation_ratio*100:.1f}%",
                timestamp_us=data.timestamp_us,
                context={
                    'dp_group_id': dp_group_id,
                    'group_average': group_average,
                    'overall_mean': overall_mean,
                    'overall_std': overall_std,
                    'deviation_ratio': deviation_ratio,
                    'z_score': z_score,
                    'num_groups': len(all_group_averages)
                }
            )
        
        return None
    
    def update_baseline(self, data: StructuredData):
        """Update baseline (DP group comparison doesn't need baseline update)"""
        pass

