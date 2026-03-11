"""
Neural Network Models for Fraud Detection
==========================================
Deep neural networks for fraud detection offer:
- Learning complex non-linear patterns
- Automatic feature interaction discovery
- End-to-end learning from raw features
- Scalability to large datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                           precision_recall_curve, f1_score)
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import joblib
import json

logger = logging.getLogger(__name__)


class FraudNeuralNetwork(nn.Module):
    """
    PyTorch neural network for fraud detection.
    
    Architecture designed for fraud detection:
    - Batch normalization for stable training
    - Dropout for regularization
    - Residual connections for deep networks
    - Custom activation functions
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.3, use_batch_norm: bool = True,
                 activation: str = 'relu', use_residual: bool = False):
        """
        Initialize neural network architecture.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu')
            use_residual: Whether to use residual connections
        """
        super(FraudNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            # Dropout (increasing dropout for deeper layers)
            layer_dropout = dropout_rate * (i + 1) / len(hidden_dims)
            layers.append(nn.Dropout(layer_dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
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
        """Forward pass."""
        return self.network(x)


class FraudNNModel:
    """
    Neural network model wrapper for fraud detection.
    
    Provides a high-level interface for training, evaluating,
    and deploying neural networks for fraud detection.
    """
    
    def __init__(self, input_dim: int, config: Optional[Dict] = None,
                 device: Optional[str] = None, random_state: int = 42):
        """
        Initialize neural network model.
        
        Args:
            input_dim: Number of input features
            config: Model configuration dictionary
            device: Device to use ('cuda' or 'cpu')
            random_state: Random seed
        """
        self.input_dim = input_dim
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
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.3,
            'use_batch_norm': True,
            'activation': 'leaky_relu',
            'use_residual': False,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 512,
            'epochs': 100,
            'early_stopping_patience': 10,
            'scheduler_factor': 0.5,
            'scheduler_patience': 5,
            'optimizer': 'adamw',
            'loss_function': 'cross_entropy',
            'class_weight_method': 'balanced'
        }
        
        if config:
            self.default_config.update(config)
        
        self.config = self.default_config
        
        # Build model
        self.model = FraudNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=self.config['hidden_dims'],
            dropout_rate=self.config['dropout_rate'],
            use_batch_norm=self.config['use_batch_norm'],
            activation=self.config['activation'],
            use_residual=self.config['use_residual']
        ).to(self.device)
        
        self.feature_names = None
        self.scaler = None
        self.threshold = 0.5
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rates': []
        }
        
        logger.info(f"Neural network initialized with {self.count_parameters():,} parameters")
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _prepare_data(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None,
                     batch_size: Optional[int] = None, shuffle: bool = True) -> DataLoader:
        """
        Prepare DataLoader from numpy arrays or pandas DataFrames.
        
        Args:
            X: Features
            y: Labels (optional for prediction)
            batch_size: Batch size (uses config if None)
            shuffle: Whether to shuffle data
        
        Returns:
            PyTorch DataLoader
        """
        if batch_size is None:
            batch_size = self.config['batch_size']
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values.astype(np.float32)
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
        else:
            X_array = X.astype(np.float32)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_array).to(self.device)
        
        if y is not None:
            if isinstance(y, pd.Series):
                y_array = y.values.astype(np.int64)
            else:
                y_array = y.astype(np.int64)
            
            y_tensor = torch.LongTensor(y_array).to(self.device)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        
        # Create DataLoader
        if shuffle and y is not None:
            # Handle class imbalance with weighted sampling
            if self.config.get('use_weighted_sampler', False) and y is not None:
                class_counts = np.bincount(y_array)
                class_weights = 1.0 / class_counts
                sample_weights = class_weights[y_array]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    sampler=sampler
                )
            else:
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=shuffle
                )
        else:
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle
            )
        
        return dataloader
    
    def _get_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """
        Calculate class weights for imbalanced data.
        
        Args:
            y: Labels array
        
        Returns:
            Tensor of class weights
        """
        class_counts = np.bincount(y)
        
        if self.config['class_weight_method'] == 'balanced':
            # n_samples / (n_classes * np.bincount(y))
            weights = len(y) / (len(class_counts) * class_counts)
        elif self.config['class_weight_method'] == 'inverse':
            weights = 1.0 / class_counts
        elif self.config['class_weight_method'] == 'sqrt_inverse':
            weights = 1.0 / np.sqrt(class_counts)
        else:
            weights = np.ones(len(class_counts))
        
        # Normalize weights
        weights = weights / weights.sum() * len(class_counts)
        
        return torch.FloatTensor(weights).to(self.device)
    
    def train(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: np.ndarray,
              X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
              y_val: Optional[np.ndarray] = None,
              verbose: bool = True) -> 'FraudNNModel':
        """
        Train neural network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            verbose: Whether to print progress
        
        Returns:
            Trained model
        """
        logger.info("Starting neural network training...")
        
        # Prepare data loaders
        train_loader = self._prepare_data(X_train, y_train, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_loader = self._prepare_data(X_val, y_val, shuffle=False)
            use_validation = True
        else:
            val_loader = None
            use_validation = False
        
        # Calculate class weights for loss function
        class_weights = self._get_class_weights(y_train)
        
        # Loss function
        if self.config['loss_function'] == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif self.config['loss_function'] == 'focal_loss':
            criterion = self._focal_loss(alpha=class_weights[1], gamma=2.0)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config['scheduler_factor'],
            patience=self.config['scheduler_patience'],
            verbose=verbose
        )
        
        # Training loop
        best_val_f1 = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            
            # Validation phase
            if use_validation:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                all_val_preds = []
                all_val_probs = []
                all_val_labels = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                        
                        all_val_preds.extend(predicted.cpu().numpy())
                        all_val_probs.extend(probs.cpu().numpy()[:, 1])
                        all_val_labels.extend(batch_y.cpu().numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100 * val_correct / val_total
                
                # Calculate validation metrics
                val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
                val_precision = precision_score(all_val_labels, all_val_preds, zero_division=0)
                val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0)
                
                self.training_history['val_loss'].append(avg_val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                self.training_history['val_f1'].append(val_f1)
                self.training_history['val_precision'].append(val_precision)
                self.training_history['val_recall'].append(val_recall)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                self.training_history['learning_rates'].append(current_lr)
                
                # Early stopping based on F1 score
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
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
                        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                        f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                        f"Val F1: {val_f1:.4f}, LR: {current_lr:.6f}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{self.config['epochs']}] - "
                        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%"
                    )
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with validation F1: {best_val_f1:.4f}")
        
        return self
    
    def _focal_loss(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Focal Loss for handling class imbalance.
        
        Focal loss focuses training on hard examples and reduces
        the loss contribution from easy examples.
        
        Args:
            alpha: Weighting factor for class 1
            gamma: Focusing parameter
        """
        class FocalLoss(nn.Module):
            def __init__(self, alpha, gamma):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
            
            def forward(self, inputs, targets):
                ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss(alpha, gamma)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict fraud probabilities.
        
        Args:
            X: Input features
        
        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        self.model.eval()
        
        dataloader = self._prepare_data(X, batch_size=self.config['batch_size'] * 2, shuffle=False)
        
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 1:
                    batch_X = batch[0]
                else:
                    batch_X = batch[0]
                
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.vstack(all_probs)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary labels."""
        if threshold is None:
            threshold = self.threshold
        
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
    
    def find_optimal_threshold(self, X_val: Union[pd.DataFrame, np.ndarray],
                                y_val: np.ndarray, metric: str = 'f1') -> float:
        """Find optimal classification threshold."""
        probs = self.predict_proba(X_val)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        
        for threshold in thresholds:
            preds = (probs >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, preds, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, preds, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, preds, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        self.threshold = thresholds[optimal_idx]
        
        logger.info(f"Optimal threshold: {self.threshold:.3f}")
        return self.threshold
    
    def evaluate(self, X_test: Union[pd.DataFrame, np.ndarray],
                 y_test: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        if threshold is None:
            threshold = self.threshold
        
        probs = self.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        metrics = {}
        
        # Basic metrics
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        metrics['accuracy'] = accuracy_score(y_test, preds)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, preds)
        metrics['precision'] = precision_score(y_test, preds, zero_division=0)
        metrics['recall'] = recall_score(y_test, preds, zero_division=0)
        metrics['f1_score'] = f1_score(y_test, preds, zero_division=0)
        
        # AUC metrics
        metrics['roc_auc'] = roc_auc_score(y_test, probs)
        metrics['pr_auc'] = average_precision_score(y_test, probs)
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'training_history': self.training_history,
            'input_dim': self.input_dim,
            'random_state': self.random_state,
            'saved_at': datetime.now().isoformat()
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'FraudNNModel':
        """Load model from disk."""
        model_data = torch.load(filepath, map_location=self.device)
        
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.threshold = model_data['threshold']
        self.training_history = model_data['training_history']
        self.input_dim = model_data['input_dim']
        self.random_state = model_data['random_state']
        
        # Rebuild model
        self.model = FraudNeuralNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.config['hidden_dims'],
            dropout_rate=self.config['dropout_rate'],
            use_batch_norm=self.config['use_batch_norm'],
            activation=self.config['activation'],
            use_residual=self.config['use_residual']
        ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        return self