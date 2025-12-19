"""
Alert manager - manages anomaly alerts
"""

from typing import Dict, List
from datetime import datetime, timedelta
import sys
import os
import uuid

sys.path.append(os.path.dirname(__file__))
from common.data_structures import AnomalyResult, Alert


class AlertManager:
    """Alert manager for handling anomaly alerts"""
    
    def __init__(self, config: dict):
        """
        Initialize alert manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.alert_levels = {
            'critical': 3,
            'warning': 2,
            'info': 1
        }
        self.deduplication_window = config.get('dedup_window', 300)  # seconds
        self.recent_alerts: Dict[str, datetime] = {}  # alert_key -> timestamp
        self.alert_level_mapping = config.get('levels', {
            'critical': ['R1', 'R4'],
            'warning': ['R2', 'R3'],
            'info': ['R7', 'R10']
        })
        self.alerts: List[Alert] = []  # Store alerts
        self.max_alerts = config.get('max_alerts', 10000)  # Maximum alerts to store
    
    def process_anomaly(self, anomaly: AnomalyResult):
        """
        Process anomaly and generate alert
        
        Args:
            anomaly: Anomaly result
        """
        # Determine alert level
        alert_level = self._determine_alert_level(anomaly)
        
        # Check for deduplication
        alert_key = self._generate_alert_key(anomaly)
        if self._is_duplicate(alert_key):
            return
        
        # Create alert
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            anomaly_result=anomaly,
            alert_level=alert_level,
            created_at=datetime.now(),
            acknowledged=False,
            resolved=False
        )
        
        # Store alert
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop(0)  # Remove oldest alert
        
        # Record for deduplication
        self.recent_alerts[alert_key] = datetime.now()
        
        # Clean old deduplication records
        self._clean_old_alerts()
        
        # Send alert (to visualization layer, etc.)
        self._send_alert(alert)
    
    def _determine_alert_level(self, anomaly: AnomalyResult) -> str:
        """
        Determine alert level based on anomaly
        
        Args:
            anomaly: Anomaly result
            
        Returns:
            Alert level string
        """
        # Check rule-based mapping
        rule_name = anomaly.rule_name
        for level, rules in self.alert_level_mapping.items():
            if rule_name in rules:
                return level
        
        # Use anomaly severity as fallback
        return anomaly.severity
    
    def _generate_alert_key(self, anomaly: AnomalyResult) -> str:
        """
        Generate deduplication key for alert
        
        Args:
            anomaly: Anomaly result
            
        Returns:
            Alert key string
        """
        # Generate key based on metric, node, rank, and rule
        return f"{anomaly.metric_name}_{anomaly.node_id}_{anomaly.rank_id}_{anomaly.rule_name}"
    
    def _is_duplicate(self, alert_key: str) -> bool:
        """
        Check if alert is duplicate
        
        Args:
            alert_key: Alert key
            
        Returns:
            True if duplicate, False otherwise
        """
        if alert_key not in self.recent_alerts:
            return False
        
        last_alert_time = self.recent_alerts[alert_key]
        time_diff = (datetime.now() - last_alert_time).total_seconds()
        
        return time_diff < self.deduplication_window
    
    def _clean_old_alerts(self):
        """Clean old deduplication records"""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, timestamp in self.recent_alerts.items():
            time_diff = (current_time - timestamp).total_seconds()
            if time_diff > self.deduplication_window:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.recent_alerts[key]
    
    def _send_alert(self, alert: Alert):
        """
        Send alert to visualization layer or other components
        
        Args:
            alert: Alert to send
        """
        # TODO: Implement actual alert sending (e.g., to message queue, WebSocket, etc.)
        print(f"Alert generated: {alert.alert_level} - {alert.anomaly_result.message}")
    
    def get_alerts(self, start_time: datetime = None, end_time: datetime = None,
                   level: str = None) -> List[Alert]:
        """
        Get alerts with filters
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            level: Alert level filter
            
        Returns:
            List of alerts
        """
        filtered_alerts = self.alerts
        
        if start_time:
            filtered_alerts = [a for a in filtered_alerts if a.created_at >= start_time]
        
        if end_time:
            filtered_alerts = [a for a in filtered_alerts if a.created_at <= end_time]
        
        if level:
            filtered_alerts = [a for a in filtered_alerts if a.alert_level == level]
        
        return filtered_alerts

