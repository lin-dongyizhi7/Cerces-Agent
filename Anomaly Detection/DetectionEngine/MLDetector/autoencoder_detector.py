"""
AutoEncoder detector
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
        
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.history: List[StructuredData] = []
        self.max_history = config.get('max_history', 1000)
        self.min_samples = config.get('min_samples', 200)  # Minimum samples before detection
        self.is_trained = False
        self.reconstruction_errors = []
        self.threshold = None
    
    def get_name(self) -> str:
        return "AutoEncoderDetector"
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute AutoEncoder detection"""
        if not self.enabled or not TORCH_AVAILABLE:
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
        X = torch.FloatTensor([features])
        
        # Reconstruct
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X)
            reconstruction_error = torch.mean((X - reconstructed) ** 2).item()
        
        # Calculate anomaly score
        if self.threshold is None:
            return None
        
        normalized_score = min(1.0, reconstruction_error / (self.threshold * 2))
        
        if reconstruction_error > self.threshold:
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=self.threshold,
                rule_name='AutoEncoder',
                detector_name=self.get_name(),
                anomaly_score=normalized_score,
                severity='critical' if normalized_score > 0.7 else 'warning',
                message=f"AutoEncoder detected anomaly. Reconstruction error: {reconstruction_error:.3f}, "
                       f"Threshold: {self.threshold:.3f}",
                timestamp_us=data.timestamp_us,
                context={
                    'reconstruction_error': reconstruction_error,
                    'threshold': self.threshold,
                    'normalized_score': normalized_score
                }
            )
        
        return None
    
    def _train_model(self):
        """Train AutoEncoder model"""
        if not TORCH_AVAILABLE or len(self.history) < self.min_samples:
            return
        
        # Extract features for all history
        X = []
        for i, data in enumerate(self.history):
            features = self.feature_extractor.extract(data, self.history[:i])
            X.append(features)
        
        X = np.array(X)
        X_tensor = torch.FloatTensor(X)
        
        # Initialize model
        input_dim = len(X[0])
        self.model = AutoEncoder(input_dim, self.hidden_dim, self.encoding_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Train
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            reconstructed = self.model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()
        
        # Calculate reconstruction errors and threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
        
        self.reconstruction_errors = errors.tolist()
        self.threshold = np.percentile(errors, self.threshold_percentile)
        self.is_trained = True
    
    def update_baseline(self, data: StructuredData):
        """Update baseline (add to history and retrain if needed)"""
        self.history.append(data)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Retrain periodically
        if len(self.history) >= self.min_samples and len(self.history) % 200 == 0:
            self._train_model()

