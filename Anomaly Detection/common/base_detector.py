"""
Base detector abstract class
"""

from abc import ABC, abstractmethod
from typing import Optional
from .data_structures import StructuredData, AnomalyResult


class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    def __init__(self, config: dict):
        """Initialize detector with configuration"""
        self.config = config
        self.enabled = config.get('enabled', True)
        self.name = self.get_name()
    
    @abstractmethod
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """
        Execute detection, return anomaly result or None
        
        Args:
            data: Structured data to detect
            
        Returns:
            AnomalyResult if anomaly detected, None otherwise
        """
        pass
    
    @abstractmethod
    def update_baseline(self, data: StructuredData):
        """
        Update baseline data
        
        Args:
            data: Structured data for baseline update
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get detector name
        
        Returns:
            Detector name
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if detector is enabled"""
        return self.enabled

