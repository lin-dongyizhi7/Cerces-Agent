"""
AutoEncoder detector
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
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, AutoEncoderDetector will not work")


class AutoEncoder(nn.Module):
    """Simple autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, encoding_dim: int = 16):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoderDetector(BaseDetector):
    """AutoEncoder detector"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        if not TORCH_AVAILABLE:
            self.enabled = False
            return
        
        self.ae_config = config.get('autoencoder', {})
        self.hidden_dim = self.ae_config.get('hidden_dim', 32)
        self.encoding_dim = self.ae_config.get('encoding_dim', 16)
        self.learning_rate = self.ae_config.get('learning_rate', 0.001)
        self.epochs = self.ae_config.get('epochs', 50)
        self.threshold_percentile = self.ae_config.get('threshold_percentile', 95)
        
        # 每个指标独立维护模型与历史
        self.models: Dict[str, AutoEncoder] = {}
        self.feature_extractor = FeatureExtractor()
        self.history: Dict[str, List[StructuredData]] = {}
        self.max_history = config.get('max_history', 1000)
        self.min_samples = config.get('min_samples', 200)  # Minimum samples before detection
        self.is_trained: Dict[str, bool] = {}
        self.reconstruction_errors: Dict[str, List[float]] = {}
        self.thresholds: Dict[str, float] = {}
    
    def get_name(self) -> str:
        return "AutoEncoderDetector"

    def _get_metric_key(self, data: StructuredData) -> str:
        """
        Build a stable key for one metric time series.
        """
        return f"{data.metric_type}|{data.metric_name}|{data.node_id}|{data.rank_id}"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute AutoEncoder detection"""
        if not self.enabled or not TORCH_AVAILABLE:
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

        if (not self.is_trained.get(metric_key) or
                self.models.get(metric_key) is None or
                metric_key not in self.thresholds):
            return None

        model = self.models[metric_key]
        threshold = self.thresholds[metric_key]
        # Extract features
        features = self.feature_extractor.extract(data, metric_history)
        X = torch.FloatTensor([features])
        
        # Reconstruct
        model.eval()
        with torch.no_grad():
            reconstructed = model(X)
            reconstruction_error = torch.mean((X - reconstructed) ** 2).item()
        
        # Calculate anomaly score
        normalized_score = min(1.0, reconstruction_error / (threshold * 2))
        
        if reconstruction_error > threshold:
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=threshold,
                rule_name='AutoEncoder',
                detector_name=self.get_name(),
                anomaly_score=normalized_score,
                severity='critical' if normalized_score > 0.7 else 'warning',
                message=f"AutoEncoder detected anomaly. Reconstruction error: {reconstruction_error:.3f}, "
                       f"Threshold: {threshold:.3f}",
                timestamp_us=data.timestamp_us,
                context={
                    'reconstruction_error': reconstruction_error,
                    'threshold': threshold,
                    'normalized_score': normalized_score
                }
            )
        
        return None
    
    def _train_model(self, metric_key: str):
        """Train AutoEncoder model for a specific metric"""
        if (not TORCH_AVAILABLE or
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
        X_tensor = torch.FloatTensor(X)
        
        # Initialize model
        input_dim = len(X[0])
        model = AutoEncoder(input_dim, self.hidden_dim, self.encoding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Train
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            reconstructed = model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()
        
        # Calculate reconstruction errors and threshold
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()

        self.reconstruction_errors[metric_key] = errors.tolist()
        self.thresholds[metric_key] = float(np.percentile(errors, self.threshold_percentile))
        self.models[metric_key] = model
        self.is_trained[metric_key] = True
    
    def update_baseline(self, data: StructuredData):
        """Update baseline (add to history and retrain if needed)"""
        if not self.enabled or not TORCH_AVAILABLE:
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
                len(metric_history) % 200 == 0):
            self._train_model(metric_key)

