"""
Rank communication detector (R5 rule)
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


class RankCommunicationDetector(BaseDetector):
    """Rank communication detector for R5 rule (intra-group rank comparison)"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.r5_config = config.get('R5', {})
        self.enabled_r5 = self.r5_config.get('enabled', True)
        self.threshold_ratio = self.r5_config.get('threshold_ratio', 0.3)  # 30% deviation threshold
        # Store rank values by timestamp: timestamp -> {rank_id: value}
        self.rank_data_cache: Dict[int, Dict[str, float]] = defaultdict(dict)
        # Store DP group info: rank_id -> dp_group_id
        self.rank_to_dp_group: Dict[str, str] = {}
        self.cache_timeout = 5  # seconds - time window for rank comparison
        self.cache_timeout_us = self.cache_timeout * 1_000_000
    
    def get_name(self) -> str:
        return "RankCommunicationDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute rank communication detection (R5 rule)"""
        if not self.enabled or not self.enabled_r5:
            return None
        
        # R5 applies to G1-G4 metrics (communication functions)
        if data.metric_type not in ['G1', 'G2', 'G3', 'G4']:
            return None
        
        # Extract DP group from metadata
        dp_group_id = data.metadata.get('dp_group_id', 'default')
        self.rank_to_dp_group[data.rank_id] = dp_group_id
        
        # Store rank data in cache
        timestamp_key = data.timestamp_us // 1_000_000  # Round to second
        self.rank_data_cache[timestamp_key][data.rank_id] = data.value
        
        # Clean old cache entries
        current_timestamp_key = data.timestamp_us // 1_000_000
        keys_to_remove = [k for k in self.rank_data_cache.keys() 
                         if current_timestamp_key - k > self.cache_timeout]
        for k in keys_to_remove:
            del self.rank_data_cache[k]
        
        # Get ranks in the same DP group
        same_group_ranks = [rid for rid, dpg in self.rank_to_dp_group.items() 
                           if dpg == dp_group_id]
        
        if len(same_group_ranks) < 2:
            return None
        
        # Get values for all ranks in the same group at this timestamp
        rank_values = []
        for rank_id in same_group_ranks:
            if rank_id in self.rank_data_cache[timestamp_key]:
                rank_values.append(self.rank_data_cache[timestamp_key][rank_id])
        
        if len(rank_values) < 2:
            return None
        
        # Calculate statistics for the group
        group_mean = statistics.mean(rank_values)
        group_std = statistics.stdev(rank_values) if len(rank_values) > 1 else 0
        
        if group_mean == 0:
            return None
        
        # Check if current rank's value deviates significantly from group mean
        deviation_ratio = abs(data.value - group_mean) / group_mean if group_mean > 0 else 0
        
        if deviation_ratio > self.threshold_ratio:
            # Calculate anomaly score
            anomaly_score = min(1.0, deviation_ratio / (self.threshold_ratio * 2))
            
            z_score = (data.value - group_mean) / group_std if group_std > 0 else 0
            
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=group_mean * (1 - self.threshold_ratio) if data.value < group_mean 
                        else group_mean * (1 + self.threshold_ratio),
                rule_name='R5',
                detector_name=self.get_name(),
                anomaly_score=anomaly_score,
                severity='critical' if deviation_ratio > 0.5 else 'warning',
                message=f"R5: Rank {data.rank_id} communication bandwidth deviates from group. "
                       f"Rank value: {data.value:.2f}, Group mean: {group_mean:.2f}, "
                       f"Deviation: {deviation_ratio*100:.1f}%",
                timestamp_us=data.timestamp_us,
                context={
                    'group_mean': group_mean,
                    'group_std': group_std,
                    'deviation_ratio': deviation_ratio,
                    'z_score': z_score,
                    'dp_group_id': dp_group_id,
                    'group_size': len(rank_values)
                }
            )
        
        return None
    
    def update_baseline(self, data: StructuredData):
        """Update baseline (rank comparison does not need baseline update)"""
        pass


