"""
Anomaly detection controller - main controller for anomaly detection layer
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))
from DetectionEngine.detection_engine import DetectionEngine
from RuleOrchestrator import RuleOrchestrator
from AlertManager import AlertManager
from DataReceiver import DataReceiver
from common.data_structures import StructuredData


class AnomalyDetectionController:
    """Main controller for anomaly detection layer"""
    
    def __init__(self, config: dict):
        """
        Initialize anomaly detection controller
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        detection_layer_config = config.get('detection_layer', {})
        
        self.data_receiver = DataReceiver(detection_layer_config.get('data_receiver', {}))
        self.detection_engine = DetectionEngine(detection_layer_config.get('detection_engine', {}))
        self.rule_orchestrator = RuleOrchestrator(detection_layer_config.get('rule_orchestrator', {}))
        self.alert_manager = AlertManager(detection_layer_config.get('alert', {}))
        
        # Setup data flow
        self._setup_data_flow()
    
    def _setup_data_flow(self):
        """Setup data flow: receive -> detect -> alert"""
        self.data_receiver.register_callback(self._on_data_received)
    
    def _on_data_received(self, data: StructuredData):
        """
        Handle received data
        
        Args:
            data: Structured data
        """
        try:
            # Get applicable detectors from rule orchestrator
            applicable_detectors = self.rule_orchestrator.get_applicable_detectors(data)
            
            # Execute detection with applicable detectors
            anomaly_results = []
            for detector_name in applicable_detectors:
                result = self.detection_engine.detect_with_detector(detector_name, data)
                if result:
                    anomaly_results.append(result)
            
            # Process anomaly results
            for anomaly in anomaly_results:
                self.alert_manager.process_anomaly(anomaly)
            
            # Update baseline for all detectors
            self.detection_engine.update_baseline(data)
        
        except Exception as e:
            print(f"Error processing data: {e}")
    
    def start(self):
        """Start anomaly detection layer"""
        self.data_receiver.start()
        print("AnomalyDetectionController started")
    
    def stop(self):
        """Stop anomaly detection layer"""
        self.data_receiver.stop()
        print("AnomalyDetectionController stopped")

