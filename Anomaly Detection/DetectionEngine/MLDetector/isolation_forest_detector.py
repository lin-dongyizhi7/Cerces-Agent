"""
Isolation Forest detector
"""

from typing import Optional, List, Dict
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
from .feature_extractor import FeatureExtractor
import uuid

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, IsolationForestDetector will not work")


class IsolationForestDetector(BaseDetector):
    """Isolation Forest detector"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        if not SKLEARN_AVAILABLE:
            self.enabled = False
            return
        
        self.if_config = config.get('isolation_forest', {})
        self.n_estimators = self.if_config.get('n_estimators', 100)
        self.contamination = self.if_config.get('contamination', 0.1)
        self.random_state = self.if_config.get('random_state', 42)
        
        # 每个指标独立维护模型与历史，避免不同指标之间的相互干扰
        self.models: Dict[str, IsolationForest] = {}
        self.feature_extractor = FeatureExtractor()
        self.history: Dict[str, List[StructuredData]] = {}
        self.max_history = config.get('max_history', 1000)
        self.min_samples = config.get('min_samples', 100)  # Minimum samples before detection
        self.is_trained: Dict[str, bool] = {}
    
    def get_name(self) -> str:
        return "IsolationForestDetector"

    def _get_metric_key(self, data: StructuredData) -> str:
        """
        Build a stable key for one metric time series.

        Using metric_type/metric_name/node_id/rank_id makes each metric stream independent.
        """
        return f"{data.metric_type}|{data.metric_name}|{data.node_id}|{data.rank_id}"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute Isolation Forest detection"""
        if not self.enabled or not SKLEARN_AVAILABLE:
            return None

        metric_key = self._get_metric_key(data)
        # Initialize containers for this metric if not exists
        if metric_key not in self.history:
            self.history[metric_key] = []
            self.is_trained[metric_key] = False

        # Add to history for this metric
        metric_history = self.history[metric_key]
        metric_history.append(data)
        if len(metric_history) > self.max_history:
            metric_history.pop(0)

        # Train model for this metric if we have enough samples
        if (not self.is_trained[metric_key] and
                len(metric_history) >= self.min_samples):
            self._train_model(metric_key)

        if not self.is_trained.get(metric_key) or self.models.get(metric_key) is None:
            return None

        model = self.models[metric_key]
        # Extract features using history of this metric only
        features = self.feature_extractor.extract(data, metric_history)
        X = np.array([features])
        
        # Predict
        prediction = model.predict(X)[0]  # -1 for anomaly, 1 for normal
        anomaly_score = -model.score_samples(X)[0]  # Negative score (higher = more anomalous)
        
        # Normalize score to 0-1
        normalized_score = min(1.0, max(0.0, (anomaly_score + 0.5) / 1.0))
        
        if prediction == -1 or normalized_score > 0.5:
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=None,
                rule_name='IsolationForest',
                detector_name=self.get_name(),
                anomaly_score=normalized_score,
                severity='critical' if normalized_score > 0.7 else 'warning',
                message=f"Isolation Forest detected anomaly. Score: {normalized_score:.3f}",
                timestamp_us=data.timestamp_us,
                context={
                    'anomaly_score': anomaly_score,
                    'normalized_score': normalized_score,
                    'prediction': int(prediction)
                }
            )
        
        return None
    
    def _train_model(self, metric_key: str):
        """Train Isolation Forest model for a specific metric"""
        if (not SKLEARN_AVAILABLE or
                metric_key not in self.history or
                len(self.history[metric_key]) < self.min_samples):
            return
        
        # Extract features for all history
        X = []
        metric_history = self.history[metric_key]
        for i, data in enumerate(metric_history):
            features = self.feature_extractor.extract(data, metric_history[:i])
            X.append(features)
        
        X = np.array(X)
        
        # Train model
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        model.fit(X)
        self.models[metric_key] = model
        self.is_trained[metric_key] = True
    
    def update_baseline(self, data: StructuredData):
        """Update baseline (add to history and retrain if needed)"""
        if not self.enabled or not SKLEARN_AVAILABLE:
            return

        metric_key = self._get_metric_key(data)
        if metric_key not in self.history:
            self.history[metric_key] = []
            self.is_trained[metric_key] = False

        metric_history = self.history[metric_key]
        metric_history.append(data)
        if len(metric_history) > self.max_history:
            metric_history.pop(0)

        # Retrain periodically for this metric
        if (len(metric_history) >= self.min_samples and
                len(metric_history) % 100 == 0):
            self._train_model(metric_key)

