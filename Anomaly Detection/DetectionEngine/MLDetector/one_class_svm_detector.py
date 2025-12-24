"""
One-Class SVM detector
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
    from sklearn.svm import OneClassSVM
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, OneClassSVMDetector will not work")


class OneClassSVMDetector(BaseDetector):
    """One-Class SVM detector"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        if not SKLEARN_AVAILABLE:
            self.enabled = False
            return
        
        self.svm_config = config.get('one_class_svm', {})
        self.nu = self.svm_config.get('nu', 0.1)  # Upper bound on fraction of outliers
        self.kernel = self.svm_config.get('kernel', 'rbf')
        self.gamma = self.svm_config.get('gamma', 'scale')
        
        # 每个指标独立维护模型与历史
        self.models: Dict[str, OneClassSVM] = {}
        self.feature_extractor = FeatureExtractor()
        self.history: Dict[str, List[StructuredData]] = {}
        self.max_history = config.get('max_history', 1000)
        self.min_samples = config.get('min_samples', 100)  # Minimum samples before detection
        self.is_trained: Dict[str, bool] = {}
    
    def get_name(self) -> str:
        return "OneClassSVMDetector"

    def _get_metric_key(self, data: StructuredData) -> str:
        """
        Build a stable key for one metric time series.
        """
        return f"{data.metric_type}|{data.metric_name}|{data.node_id}|{data.rank_id}"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute One-Class SVM detection"""
        if not self.enabled or not SKLEARN_AVAILABLE:
            return None

        metric_key = self._get_metric_key(data)
        if metric_key not in self.history:
            self.history[metric_key] = []
            self.is_trained[metric_key] = False

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
        decision_score = model.decision_function(X)[0]  # Distance to decision boundary
        
        # Normalize score to 0-1 (negative = anomaly, positive = normal)
        normalized_score = min(1.0, max(0.0, (-decision_score + 1.0) / 2.0))
        
        if prediction == -1 or normalized_score > 0.5:
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=None,
                rule_name='OneClassSVM',
                detector_name=self.get_name(),
                anomaly_score=normalized_score,
                severity='critical' if normalized_score > 0.7 else 'warning',
                message=f"One-Class SVM detected anomaly. Score: {normalized_score:.3f}",
                timestamp_us=data.timestamp_us,
                context={
                    'decision_score': float(decision_score),
                    'normalized_score': normalized_score,
                    'prediction': int(prediction)
                }
            )
        
        return None
    
    def _train_model(self, metric_key: str):
        """Train One-Class SVM model for a specific metric"""
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
        model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma
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

