"""
Transformer detector for time series anomaly detection
"""

from typing import Optional, List, Dict
import sys
import os
import numpy as np
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from common.base_detector import BaseDetector
from common.data_structures import StructuredData, AnomalyResult
import uuid

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, TransformerDetector will not work")


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, input_dim: int = 1, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch, input_dim)
        """
        # Project input to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Transpose for transformer: (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        
        # Take the last output
        last_output = encoded[-1, :, :]  # (batch, d_model)
        
        # Output projection
        output = self.output_projection(last_output)  # (batch, input_dim)
        
        return output


class TransformerDetector(BaseDetector):
    """Transformer detector for time series anomaly detection"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        if not TORCH_AVAILABLE:
            self.enabled = False
            return
        
        self.transformer_config = config.get('transformer', {})
        self.d_model = self.transformer_config.get('d_model', 64)
        self.nhead = self.transformer_config.get('nhead', 4)
        self.num_layers = self.transformer_config.get('num_layers', 2)
        self.dim_feedforward = self.transformer_config.get('dim_feedforward', 256)
        self.dropout = self.transformer_config.get('dropout', 0.1)
        self.learning_rate = self.transformer_config.get('learning_rate', 0.0001)
        self.epochs = self.transformer_config.get('epochs', 50)
        self.sequence_length = self.transformer_config.get('sequence_length', 30)  # Input sequence length
        self.threshold_percentile = self.transformer_config.get('threshold_percentile', 95)
        
        # 每个指标独立维护模型与历史
        self.models: Dict[str, TransformerModel] = {}
        self.history: Dict[str, List[float]] = {}  # Store values only for sequence construction
        self.max_history = config.get('max_history', 1000)
        self.min_samples = config.get('min_samples', 300)  # Minimum samples before detection
        self.is_trained: Dict[str, bool] = {}
        self.prediction_errors: Dict[str, List[float]] = {}  # Store prediction errors
        self.thresholds: Dict[str, float] = {}  # Threshold for each metric
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_name(self) -> str:
        return "TransformerDetector"
    
    def _get_metric_key(self, data: StructuredData) -> str:
        """Build a stable key for one metric time series"""
        return f"{data.metric_type}|{data.metric_name}|{data.node_id}|{data.rank_id}"
    
    def _create_sequences(self, values: List[float], seq_length: int) -> tuple:
        """
        Create sequences for Transformer training/prediction
        
        Args:
            values: List of values
            seq_length: Sequence length
            
        Returns:
            X: Input sequences (num_sequences, seq_length, 1)
            y: Target values (num_sequences, 1)
        """
        if len(values) < seq_length + 1:
            return None, None
        
        X, y = [], []
        for i in range(len(values) - seq_length):
            X.append(values[i:i + seq_length])
            y.append(values[i + seq_length])
        
        # Normalize sequences (z-score normalization per sequence)
        X_normalized = []
        y_normalized = []
        for seq, target in zip(X, y):
            seq_mean = np.mean(seq)
            seq_std = np.std(seq) if np.std(seq) > 1e-8 else 1.0
            X_normalized.append([(x - seq_mean) / seq_std for x in seq])
            y_normalized.append((target - seq_mean) / seq_std)
        
        X_tensor = torch.FloatTensor(X_normalized).unsqueeze(-1)  # (num_sequences, seq_length, 1)
        y_tensor = torch.FloatTensor(y_normalized).unsqueeze(-1)  # (num_sequences, 1)
        
        return X_tensor, y_tensor
    
    def detect(self, data: StructuredData) -> Optional[AnomalyResult]:
        """Execute Transformer detection"""
        if not self.enabled or not TORCH_AVAILABLE:
            return None
        
        metric_key = self._get_metric_key(data)
        
        # Initialize containers for this metric if not exists
        if metric_key not in self.history:
            self.history[metric_key] = []
            self.is_trained[metric_key] = False
            self.prediction_errors[metric_key] = []
        
        # Add to history
        metric_history = self.history[metric_key]
        metric_history.append(data.value)
        if len(metric_history) > self.max_history:
            metric_history.pop(0)
        
        # Train model if we have enough samples
        if (not self.is_trained[metric_key] and
                len(metric_history) >= self.min_samples):
            self._train_model(metric_key)
        
        if not self.is_trained.get(metric_key) or self.models.get(metric_key) is None:
            return None
        
        if len(metric_history) < self.sequence_length + 1:
            return None
        
        # Get recent sequence for prediction
        recent_values = metric_history[-self.sequence_length:]
        seq_mean = np.mean(recent_values)
        seq_std = np.std(recent_values) if np.std(recent_values) > 1e-8 else 1.0
        
        # Normalize sequence
        normalized_seq = [(x - seq_mean) / seq_std for x in recent_values]
        X = torch.FloatTensor([normalized_seq]).unsqueeze(-1).to(self.device)  # (1, seq_length, 1)
        
        # Predict
        model = self.models[metric_key]
        model.eval()
        with torch.no_grad():
            prediction_normalized = model(X)
            prediction = prediction_normalized.item() * seq_std + seq_mean
        
        # Calculate prediction error
        actual_value = data.value
        prediction_error = abs(actual_value - prediction)
        
        # Normalize error by sequence std for fair comparison
        normalized_error = prediction_error / seq_std if seq_std > 1e-8 else prediction_error
        
        # Update error history
        error_history = self.prediction_errors[metric_key]
        error_history.append(normalized_error)
        if len(error_history) > self.max_history:
            error_history.pop(0)
        
        # Check threshold
        if metric_key not in self.thresholds:
            return None
        
        threshold = self.thresholds[metric_key]
        
        if normalized_error > threshold:
            # Calculate anomaly score
            anomaly_score = min(1.0, normalized_error / (threshold * 2))
            
            return AnomalyResult(
                anomaly_id=str(uuid.uuid4()),
                metric_name=data.metric_name,
                metric_type=data.metric_type,
                node_id=data.node_id,
                rank_id=data.rank_id,
                value=data.value,
                threshold=prediction + threshold * seq_std if seq_std > 1e-8 else prediction + threshold,
                rule_name='Transformer',
                detector_name=self.get_name(),
                anomaly_score=anomaly_score,
                severity='critical' if anomaly_score > 0.7 else 'warning',
                message=f"Transformer detected anomaly. Prediction: {prediction:.2f}, Actual: {actual_value:.2f}, "
                       f"Error: {normalized_error:.3f}, Threshold: {threshold:.3f}",
                timestamp_us=data.timestamp_us,
                context={
                    'prediction': float(prediction),
                    'prediction_error': float(prediction_error),
                    'normalized_error': float(normalized_error),
                    'threshold': float(threshold)
                }
            )
        
        return None
    
    def _train_model(self, metric_key: str):
        """Train Transformer model for a specific metric"""
        if (not TORCH_AVAILABLE or
                metric_key not in self.history or
                len(self.history[metric_key]) < self.min_samples):
            return
        
        values = self.history[metric_key]
        
        # Create sequences
        X, y = self._create_sequences(values, self.sequence_length)
        if X is None or y is None:
            return
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Initialize model
        model = TransformerModel(
            input_dim=1,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Train
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Calculate prediction errors on training data
        model.eval()
        with torch.no_grad():
            predictions = model(X)
            errors = torch.abs(predictions - y).cpu().numpy().flatten()
        
        # Store errors and calculate threshold
        self.prediction_errors[metric_key] = errors.tolist()
        self.thresholds[metric_key] = np.percentile(errors, self.threshold_percentile)
        
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
            self.prediction_errors[metric_key] = []
        
        metric_history = self.history[metric_key]
        metric_history.append(data.value)
        if len(metric_history) > self.max_history:
            metric_history.pop(0)
        
        # Retrain periodically
        if (len(metric_history) >= self.min_samples and
                len(metric_history) % 300 == 0):
            self._train_model(metric_key)

