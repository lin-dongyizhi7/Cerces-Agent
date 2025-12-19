"""
LOF (Local Outlier Factor) detector
"""

from typing import Optional, List
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
from .feature_extractor import FeatureExtractor
import uuid

try:
    from sklearn.neighbors import LocalOutlierFactor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, LOFDetector will not work")


class LOFDetector(BaseDetector):
    """Local Outlier Factor detector"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        if not SKLEARN_AVAILABLE:
            self.enabled = False
            return
        
        self.lof_config = config.get('lof', {})
        self.n_neighbors = self.lof_config.get('n_neighbors', 20)
        self.contamination = self.lof_config.get('contamination', 0.1)
        
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.history: List[StructuredData] = []
        self.max_history = config.get('max_history', 1000)
        self.min_samples = config.get('min_samples', 50)  # Minimum samples before detection
        self.is_trained = False
    
    def get_name(self) -> str:
        return "LOFDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute LOF detection"""
        if not self.enabled or not SKLEARN_AVAILABLE:
            return None
        
        # Add to history
        self.history.append(data)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Train model if we have enough samples
        if not self.is_trained and len(self.history) >= self.min_samples:
            self._train_model()
        
        if not self.is_trained or self.model is None:
            return None
        
        # Extract features
        features = self.feature_extractor.extract(data, self.history)
        X = np.array([features])
        
        # Predict
        prediction = self.model.predict(X)[0]  # -1 for anomaly, 1 for normal
        anomaly_score = -self.model.score_samples(X)[0]  # Negative LOF score
        
        # Normalize score to 0-1 (LOF typically ranges from 0.5 to 2.0)
        normalized_score = min(1.0, max(0.0, (anomaly_score - 0.5) / 1.5))
        
        if prediction == -1 or normalized_score > 0.5:
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=None,
                rule_name='LOF',
                detector_name=self.get_name(),
                anomaly_score=normalized_score,
                severity='critical' if normalized_score > 0.7 else 'warning',
                message=f"LOF detected anomaly. Score: {normalized_score:.3f}",
                timestamp_us=data.timestamp_us,
                context={
                    'anomaly_score': anomaly_score,
                    'normalized_score': normalized_score,
                    'prediction': int(prediction),
                    'n_neighbors': self.n_neighbors
                }
            )
        
        return None
    
    def _train_model(self):
        """Train LOF model"""
        if not SKLEARN_AVAILABLE or len(self.history) < self.min_samples:
            return
        
        # Extract features for all history
        X = []
        for i, data in enumerate(self.history):
            features = self.feature_extractor.extract(data, self.history[:i])
            X.append(features)
        
        X = np.array(X)
        
        # Train model
        self.model = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(X) - 1),
            contamination=self.contamination,
            novelty=True
        )
        self.model.fit(X)
        self.is_trained = True
    
    def update_baseline(self, data: StructuredData):
        """Update baseline (add to history and retrain if needed)"""
        self.history.append(data)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Retrain periodically
        if len(self.history) >= self.min_samples and len(self.history) % 100 == 0:
            self._train_model()

