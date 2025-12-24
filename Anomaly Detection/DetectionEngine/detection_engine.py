"""
Detection engine - manages multiple detectors
"""

from typing import Optional, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
from StatisticalDetector.statistical_detector import StatisticalDetector
from MLDetector.ml_detector import MLDetector


class DetectionEngine:
    """Detection engine that manages multiple detectors"""
    
    def __init__(self, config: dict):
        """
        Initialize detection engine with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.detectors: Dict[str, BaseDetector] = {}
        self._initialize_detectors(config)
    
    def _initialize_detectors(self, config: dict):
        """Initialize all detectors based on configuration"""
        # Unified statistical detector (includes comparison-based statistical methods)
        if config.get('statistical', {}).get('enabled', True) or config.get('comparison', {}).get('enabled', True):
            statistical_config = dict(config)
            self.detectors['statistical'] = StatisticalDetector(statistical_config)

        # ML detector
        if config.get('ml', {}).get('enabled', True):
            ml_config = config.get('ml', {})
            ml_config.update(config)  # Merge parent config
            self.detectors['ml'] = MLDetector(ml_config)
    
    def detect_with_detector(self, detector_name: str, 
                            data: StructuredData) -> Optional[AnomalyResult]:
        """
        Execute detection using specified detector
        
        Args:
            detector_name: Name of detector ('statistical', 'ml')
            data: Structured data to detect
            
        Returns:
            AnomalyResult if anomaly detected, None otherwise
        """
        if detector_name not in self.detectors:
            return None
        
        detector = self.detectors[detector_name]
        if not detector.is_enabled():
            return None
        
        return detector.detect(data)
    
    def detect_all(self, data: StructuredData) -> Dict[str, AnomalyResult]:
        """
        Execute detection using all detectors
        
        Args:
            data: Structured data to detect
            
        Returns:
            Dictionary mapping detector names to AnomalyResults
        """
        results = {}
        for name, detector in self.detectors.items():
            if detector.is_enabled():
                result = detector.detect(data)
                if result:
                    results[name] = result
        
        return results
    
    def register_detector(self, name: str, detector: BaseDetector):
        """
        Register a new detector
        
        Args:
            name: Detector name
            detector: Detector instance
        """
        self.detectors[name] = detector
    
    def update_baseline(self, data: StructuredData):
        """
        Update baseline for all detectors
        
        Args:
            data: Structured data for baseline update
        """
        for detector in self.detectors.values():
            if detector.is_enabled():
                detector.update_baseline(data)
    
    def get_detector(self, name: str) -> Optional[BaseDetector]:
        """
        Get detector by name
        
        Args:
            name: Detector name
            
        Returns:
            Detector instance or None
        """
        return self.detectors.get(name)

