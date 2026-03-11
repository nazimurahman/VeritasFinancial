"""
Autoencoder Models for Anomaly Detection
==========================================
Autoencoders are powerful for fraud detection because:
- Learn normal transaction patterns
- Detect anomalies through reconstruction error
- Unsupervised learning (no labels needed)
- Can detect new fraud patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                           precision_recall_curve, f1_score)
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)


class FraudAutoencoder(nn.Module):
    """
    Autoencoder neural network for anomaly detection.
    
    Architecture:
    - Encoder: Compresses input to latent representation
    - Decoder: Reconstructs input from latent representation
    - Bottleneck: Forces learning of meaningful representations
    
    For fraud detection, fraudulent transactions will have
    higher reconstruction error.
    """
    
    def __init__(self, input_dim: int, encoding_dims: List[int] = [64, 32, 16],
                 activation: str = 'relu', dropout_rate: float = 0.1):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Number of input features
            encoding_dims: Dimensions of encoding layers (bottleneck is last)
            activation: Activation function
            dropout_rate: Dropout probability
        """
        super(FraudAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.bottleneck_dim = encoding_dims[-1]
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            if activation == 'relu':
                encoder_layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                encoder_layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu':
                encoder_layers.append(nn.ELU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        decoding_dims = encoding_dims[:-1][::-1] + [input_dim]
        prev_dim = encoding_dims[-1]
        
        for dim in decoding_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if dim != input_dim:  # No batch norm on output
                decoder_layers.append(nn.BatchNorm1d(dim))
            if dim != input_dim:  # No activation on output
                if activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    decoder_layers.append(nn.LeakyReLU(0.1))
                elif activation == 'elu':
                    decoder_layers.append(nn.ELU())
            if dim != input_dim:  # No dropout on output
                decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x):
        """Get encoded representation."""
        return self.encoder(x)


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for anomaly detection.
    
    VAE adds probabilistic modeling to autoencoders:
    - Learns distribution of normal data
    - Provides uncertainty estimates
    - Better generalization
    """
    
    def __init__(self, input_dim: int, encoding_dims: List[int] = [64, 32],
                 latent_dim: int = 16, activation: str = 'relu'):
        """
        Initialize VAE.
        
        Args:
            input_dim: Number of input features
            encoding_dims: Dimensions of encoding layers
            latent_dim: Dimension of latent space
            activation: Activation function
        """
        super(VariationalAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            if activation == 'relu':
                encoder_layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                encoder_layers.append(nn.LeakyReLU(0.1))
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(encoding_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoding_dims[-1], latent_dim)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        decoding_dims = encoding_dims[::-1] + [input_dim]
        
        for dim in decoding_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if dim != input_dim:  # No batch norm on output
                decoder_layers.append(nn.BatchNorm1d(dim))
            if dim != input_dim:  # No activation on output
                if activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    decoder_layers.append(nn.LeakyReLU(0.1))
            prev_dim = dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass through VAE."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z


class FraudAutoencoderModel:
    """
    Autoencoder-based anomaly detection for fraud.
    
    Uses reconstruction error to identify fraudulent transactions:
    - Train on normal transactions only
    - Fraudulent transactions have higher reconstruction error
    - Threshold on reconstruction error for classification
    """
    
    def __init__(self, input_dim: int, model_type: str = 'autoencoder',
                 config: Optional[Dict] = None, device: Optional[str] = None,
                 random_state: int = 42):
        """
        Initialize autoencoder model.
        
        Args:
            input_dim: Number of input features
            model_type: 'autoencoder' or 'vae'
            config: Model configuration
            device: Device to use
            random_state: Random seed
        """
        self.input_dim = input_dim
        self.model_type = model_type
        self.random_state = random_state
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Default configuration
        self.default_config = {
            'encoding_dims': [64, 32, 16],
            'latent_dim': 16,
            'activation': 'relu',
            'dropout_rate': 0.1,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 256,
            'epochs': 50,
            'early_stopping_patience': 10,
            'reconstruction_error_percentile': 95,
            'contamination': 0.01  # Expected fraud rate
        }
        
        if config:
            self.default_config.update(config)
        
        self.config = self.default_config
        
        # Build model
        if model_type == 'autoencoder':
            self.model = FraudAutoencoder(
                input_dim=input_dim,
                encoding_dims=self.config['encoding_dims'],
                activation=self.config['activation'],
                dropout_rate=self.config['dropout_rate']
            ).to(self.device)
        elif model_type == 'vae':
            self.model = VariationalAutoencoder(
                input_dim=input_dim,
                encoding_dims=self.config['encoding_dims'],
                latent_dim=self.config['latent_dim'],
                activation=self.config['activation']
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.scaler = StandardScaler()
        self.threshold = None
        self.reconstruction_errors = None
        self.feature_names = None
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }
        
        logger.info(f"Autoencoder initialized with {self.count_parameters():,} parameters")
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _prepare_data(self, X: Union[pd.DataFrame, np.ndarray],
                     batch_size: Optional[int] = None) -> DataLoader:
        """Prepare DataLoader for autoencoder training."""
        if batch_size is None:
            batch_size = self.config['batch_size']
        
        # Scale data
        if not hasattr(self, 'scaler_fitted'):
            X_scaled = self.scaler.fit_transform(X)
            self.scaler_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
    
    def _reconstruction_loss(self, x_recon, x, mu=None, logvar=None):
        """
        Calculate reconstruction loss.
        
        For VAE, includes KL divergence.
        """
        # MSE reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
        
        if self.model_type == 'vae' and mu is not None and logvar is not None:
            # KL divergence for VAE
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / x.size(0)  # Normalize by batch size
            return recon_loss + 0.001 * kl_loss  # Weighted combination
        
        return recon_loss
    
    def train(self, X_train: Union[pd.DataFrame, np.ndarray],
              X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
              verbose: bool = True) -> 'FraudAutoencoderModel':
        """
        Train autoencoder on normal transactions.
        
        Args:
            X_train: Training data (should contain only normal transactions)
            X_val: Validation data (optional)
            verbose: Whether to print progress
        
        Returns:
            Trained model
        """
        logger.info("Starting autoencoder training...")
        
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        # Prepare data loaders
        train_loader = self._prepare_data(X_train)
        
        if X_val is not None:
            val_loader = self._prepare_data(X_val)
            use_validation = True
        else:
            val_loader = None
            use_validation = False
        
        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                X_batch = batch[0]
                
                optimizer.zero_grad()
                
                if self.model_type == 'autoencoder':
                    X_recon, _ = self.model(X_batch)
                    loss = self._reconstruction_loss(X_recon, X_batch)
                else:  # VAE
                    X_recon, mu, logvar, _ = self.model(X_batch)
                    loss = self._reconstruction_loss(X_recon, X_batch, mu, logvar)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.training_history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if use_validation:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        X_batch = batch[0]
                        
                        if self.model_type == 'autoencoder':
                            X_recon, _ = self.model(X_batch)
                            loss = self._reconstruction_loss(X_recon, X_batch)
                        else:
                            X_recon, mu, logvar, _ = self.model(X_batch)
                            loss = self._reconstruction_loss(X_recon, X_batch, mu, logvar)
                        
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                self.training_history['val_loss'].append(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{self.config['epochs']}] - "
                        f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{self.config['epochs']}] - "
                        f"Train Loss: {avg_train_loss:.6f}"
                    )
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with validation loss: {best_val_loss:.6f}")
        
        # Calculate reconstruction errors for threshold
        self._calculate_threshold(X_train)
        
        return self
    
    def _calculate_threshold(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Calculate threshold for anomaly detection.
        
        Uses percentile of reconstruction errors on normal data.
        """
        self.model.eval()
        
        # Get reconstruction errors
        errors = self.get_reconstruction_errors(X)
        
        # Store for reference
        self.reconstruction_errors = errors
        
        # Set threshold at specified percentile
        percentile = self.config['reconstruction_error_percentile']
        self.threshold = np.percentile(errors, percentile)
        
        logger.info(f"Reconstruction error threshold ({percentile}th percentile): {self.threshold:.6f}")
        logger.info(f"Error statistics - Mean: {np.mean(errors):.6f}, Std: {np.std(errors):.6f}")
    
    def get_reconstruction_errors(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Calculate reconstruction errors for input data.
        
        Args:
            X: Input features
        
        Returns:
            Array of reconstruction errors (MSE per sample)
        """
        self.model.eval()
        
        # Prepare data
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        errors = []
        
        with torch.no_grad():
            # Process in batches
            batch_size = self.config['batch_size']
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                
                if self.model_type == 'autoencoder':
                    X_recon, _ = self.model(batch)
                else:
                    X_recon, _, _, _ = self.model(batch)
                
                # MSE per sample
                batch_errors = torch.mean((batch - X_recon) ** 2, dim=1)
                errors.extend(batch_errors.cpu().numpy())
        
        return np.array(errors)
    
    def predict_anomaly_score(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get anomaly scores (reconstruction errors).
        
        Higher score indicates more anomalous (potentially fraudulent).
        
        Args:
            X: Input features
        
        Returns:
            Array of anomaly scores
        """
        return self.get_reconstruction_errors(X)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict binary anomaly labels.
        
        Returns:
            1 for anomaly (fraud), 0 for normal
        """
        if self.threshold is None:
            raise ValueError("Model must be trained before prediction")
        
        errors = self.get_reconstruction_errors(X)
        return (errors > self.threshold).astype(int)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get probability-like anomaly scores.
        
        Normalizes reconstruction errors to [0, 1] range.
        """
        errors = self.get_reconstruction_errors(X)
        
        # Normalize to [0, 1] based on threshold
        probs = np.clip(errors / (self.threshold * 2), 0, 1)
        
        # Return 2-column format (normal prob, anomaly prob)
        return np.column_stack([1 - probs, probs])
    
    def evaluate(self, X_test: Union[pd.DataFrame, np.ndarray],
                 y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate autoencoder performance.
        
        Args:
            X_test: Test features
            y_test: True labels (1 for fraud, 0 for normal)
        
        Returns:
            Dictionary of metrics
        """
        errors = self.get_reconstruction_errors(X_test)
        
        metrics = {}
        
        # If threshold is set, evaluate classification
        if self.threshold is not None:
            preds = (errors > self.threshold).astype(int)
            
            metrics['accuracy'] = np.mean(preds == y_test)
            metrics['precision'] = precision_score(y_test, preds, zero_division=0)
            metrics['recall'] = recall_score(y_test, preds, zero_division=0)
            metrics['f1_score'] = f1_score(y_test, preds, zero_division=0)
        
        # Threshold-independent metrics
        metrics['roc_auc'] = roc_auc_score(y_test, errors)
        metrics['pr_auc'] = average_precision_score(y_test, errors)
        
        logger.info("Autoencoder evaluation complete")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{metric:15s}: {value:.4f}")
        
        return metrics
    
    def find_optimal_threshold(self, X_val: Union[pd.DataFrame, np.ndarray],
                               y_val: np.ndarray, metric: str = 'f1') -> float:
        """
        Find optimal threshold using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Optimization metric
        
        Returns:
            Optimal threshold
        """
        errors = self.get_reconstruction_errors(X_val)
        
        # Test thresholds from percentiles
        percentiles = np.arange(50, 99.5, 0.5)
        thresholds = np.percentile(errors, percentiles)
        
        best_score = -np.inf
        best_threshold = None
        
        for threshold in thresholds:
            preds = (errors > threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, preds, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, preds, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, preds, zero_division=0)
            elif metric == 'pr_auc':
                # Use PR-AUC at this threshold
                score = average_precision_score(y_val, errors > threshold)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.threshold = best_threshold
        logger.info(f"Optimal threshold: {best_threshold:.6f} (score: {best_score:.4f})")
        
        return best_threshold
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'config': self.config,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'reconstruction_errors': self.reconstruction_errors,
            'training_history': self.training_history,
            'input_dim': self.input_dim,
            'random_state': self.random_state,
            'saved_at': datetime.now().isoformat()
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'FraudAutoencoderModel':
        """Load model from disk."""
        model_data = torch.load(filepath, map_location=self.device)
        
        self.model_type = model_data['model_type']
        self.config = model_data['config']
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.feature_names = model_data['feature_names']
        self.reconstruction_errors = model_data['reconstruction_errors']
        self.training_history = model_data['training_history']
        self.input_dim = model_data['input_dim']
        self.random_state = model_data['random_state']
        
        # Rebuild model
        if self.model_type == 'autoencoder':
            self.model = FraudAutoencoder(
                input_dim=self.input_dim,
                encoding_dims=self.config['encoding_dims'],
                activation=self.config['activation'],
                dropout_rate=self.config['dropout_rate']
            ).to(self.device)
        else:
            self.model = VariationalAutoencoder(
                input_dim=self.input_dim,
                encoding_dims=self.config['encoding_dims'],
                latent_dim=self.config['latent_dim'],
                activation=self.config['activation']
            ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        return self