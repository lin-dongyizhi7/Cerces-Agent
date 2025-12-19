"""
FLOPS comparison detector (R4 rule)
"""

from typing import Optional, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
import uuid


class FLOPSComparisonDetector(BaseDetector):
    """FLOPS comparison detector for R4 rule"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.r4_config = config.get('R4', {})
        self.enabled_r4 = self.r4_config.get('enabled', True)
        self.flops_baselines = self.r4_config.get('baselines', {})
        self.threshold_ratio = self.r4_config.get('threshold_ratio', 0.15)  # 15% deviation threshold
        # Metric name to baseline mapping
        self.metric_baselines: Dict[str, float] = {}
        
        # Initialize baselines from config
        # F1: aclnnFlashAttentionScore, F2: aclnnMatmul, F3: aclnnBatchMatMul, F4: aclnnFFN
        metric_mapping = {
            'F1': 'aclnnFlashAttentionScore',
            'F2': 'aclnnMatmul',
            'F3': 'aclnnBatchMatMul',
            'F4': 'aclnnFFN'
        }
        
        for metric_type, metric_name_prefix in metric_mapping.items():
            # Try to find baseline in config
            for key, value in self.flops_baselines.items():
                if metric_name_prefix in key or metric_type in key:
                    self.metric_baselines[metric_type] = value
                    break
    
    def get_name(self) -> str:
        return "FLOPSComparisonDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute FLOPS comparison detection (R4 rule)"""
        if not self.enabled or not self.enabled_r4:
            return None
        
        # R4 applies to F1-F4 metrics (compute-intensive kernel functions)
        if data.metric_type not in ['F1', 'F2', 'F3', 'F4']:
            return None
        
        # Get baseline for this metric type
        baseline = self.metric_baselines.get(data.metric_type)
        if baseline is None:
            # Try to get from metric name
            baseline = self.flops_baselines.get(data.metric_name)
        
        if baseline is None or baseline == 0:
            return None
        
        # Check if FLOPS deviates significantly from baseline
        deviation_ratio = abs(data.value - baseline) / baseline
        
        if deviation_ratio > self.threshold_ratio:
            # Calculate anomaly score
            anomaly_score = min(1.0, deviation_ratio / (self.threshold_ratio * 2))
            
            trend = 'below' if data.value < baseline else 'above'
            
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=baseline * (1 - self.threshold_ratio) if data.value < baseline 
                        else baseline * (1 + self.threshold_ratio),
                rule_name='R4',
                detector_name=self.get_name(),
                anomaly_score=anomaly_score,
                severity='critical' if deviation_ratio > 0.3 else 'warning',
                message=f"R4: {data.metric_name} FLOPS is {trend} baseline. "
                       f"Current: {data.value:.2f}, Baseline: {baseline:.2f}, "
                       f"Deviation: {deviation_ratio*100:.1f}%",
                timestamp_us=data.timestamp_us,
                context={
                    'baseline_flops': baseline,
                    'deviation_ratio': deviation_ratio,
                    'trend': trend
                }
            )
        
        return None
    
    def update_baseline(self, data: StructuredData):
        """Update baseline FLOPS (optional, can be static)"""
        if data.metric_type in ['F1', 'F2', 'F3', 'F4']:
            if data.metric_type not in self.metric_baselines:
                self.metric_baselines[data.metric_type] = data.value
            # Optionally update baseline using exponential moving average
            # alpha = 0.05  # Very slow update
            # self.metric_baselines[data.metric_type] = (
            #     alpha * data.value + (1 - alpha) * self.metric_baselines[data.metric_type]
            # )

