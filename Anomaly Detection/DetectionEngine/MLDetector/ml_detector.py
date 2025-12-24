"""
Machine learning detector - aggregates multiple ML detection methods
"""

from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
from .isolation_forest_detector import IsolationForestDetector
from .lof_detector import LOFDetector
from .one_class_svm_detector import OneClassSVMDetector
from .autoencoder_detector import AutoEncoderDetector
from .lstm_detector import LSTMDetector
from .transformer_detector import TransformerDetector


class MLDetector(BaseDetector):
    """Machine learning detector that aggregates multiple ML detection methods"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.detectors = []
        
        ml_config = config.get('ml', {})
        methods = ml_config.get('methods', ['isolation_forest'])
        
        # Initialize ML detectors based on config
        if 'isolation_forest' in methods:
            if_config = ml_config.get('isolation_forest', {})
            if_config.update(config)
            self.detectors.append(IsolationForestDetector(if_config))
        
        if 'lof' in methods:
            lof_config = ml_config.get('lof', {})
            lof_config.update(config)
            self.detectors.append(LOFDetector(lof_config))
        
        if 'one_class_svm' in methods:
            svm_config = ml_config.get('one_class_svm', {})
            svm_config.update(config)
            self.detectors.append(OneClassSVMDetector(svm_config))
        
        if 'autoencoder' in methods:
            ae_config = ml_config.get('autoencoder', {})
            ae_config.update(config)
            self.detectors.append(AutoEncoderDetector(ae_config))
        
        if 'lstm' in methods:
            lstm_config = ml_config.get('lstm', {})
            lstm_config.update(config)
            self.detectors.append(LSTMDetector(lstm_config))
        
        if 'transformer' in methods:
            transformer_config = ml_config.get('transformer', {})
            transformer_config.update(config)
            self.detectors.append(TransformerDetector(transformer_config))
    
    def get_name(self) -> str:
        return "MLDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute ML detection using all sub-detectors"""
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

