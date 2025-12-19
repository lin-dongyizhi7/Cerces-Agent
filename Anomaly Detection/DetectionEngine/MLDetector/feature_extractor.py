"""
Feature extractor for ML detectors
"""

from typing import List, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from common.data_structures import StructuredData


class FeatureExtractor:
    """Extract features from structured data for ML models"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract(self, data: StructuredData, history: List[StructuredData] = None) -> List[float]:
        """
        Extract features from structured data
        
        Args:
            data: Current structured data
            history: Historical data for context (optional)
            
        Returns:
            Feature vector
        """
        features = []
        
        # Basic features
        features.append(data.value)
        features.append(float(data.step_id))
        features.append(float(data.timestamp_us) / 1e9)  # Convert to seconds
        
        # Metric type encoding (one-hot like)
        metric_types = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 
                       'T11', 'T12', 'T13', 'T14', 'T15', 'D1', 'F1', 'F2', 'F3', 'F4',
                       'B1', 'B2', 'B3', 'B4', 'B5', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']
        metric_type_idx = metric_types.index(data.metric_type) if data.metric_type in metric_types else -1
        features.append(float(metric_type_idx))
        
        # Rank ID encoding (simple hash)
        rank_hash = hash(data.rank_id) % 1000
        features.append(float(rank_hash))
        
        # Node ID encoding (simple hash)
        node_hash = hash(data.node_id) % 1000
        features.append(float(node_hash))
        
        # Historical features if available
        if history and len(history) > 0:
            # Mean of recent values
            recent_values = [h.value for h in history[-10:]]
            features.append(sum(recent_values) / len(recent_values))
            
            # Std of recent values
            if len(recent_values) > 1:
                mean_val = sum(recent_values) / len(recent_values)
                variance = sum((x - mean_val) ** 2 for x in recent_values) / (len(recent_values) - 1)
                features.append(variance ** 0.5)
            else:
                features.append(0.0)
            
            # Trend (slope)
            if len(recent_values) >= 2:
                trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                features.append(trend)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0])  # No history
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get feature dimension"""
        return 10  # Basic features + historical features

