"""
LSTM Models for Sequential Fraud Detection
============================================
LSTM networks are ideal for fraud detection because:
- Model sequential transaction patterns
- Capture temporal dependencies
- Learn user behavior over time
- Detect anomalies in transaction sequences
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                           f1_score, precision_score, recall_score)
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import joblib

logger = logging.getLogger(__name__)


class TransactionSequenceDataset(Dataset):
    """
    Custom dataset for transaction sequences.
    
    Creates sequences of transactions for each user:
    - Each sample is a sequence of recent transactions
    - Target is fraud label for the next transaction
    """
    
    def __init__(self, transactions_df: pd.DataFrame,
                 user_col: str = 'customer_id',
                 time_col: str = 'transaction_time',
                 feature_cols: List[str] = None,
                 target_col: str = 'is_fraud',
                 sequence_length: int = 10,
                 min_sequence_length: int = 3):
        """
        Initialize sequence dataset.
        
        Args:
            transactions_df: DataFrame with transaction data
            user_col: Column name for user ID
            time_col: Column name for timestamp
            feature_cols: List of feature columns to use
            target_col: Column name for target
            sequence_length: Length of sequences to create
            min_sequence_length: Minimum sequence length to include
        """
        self.user_col = user_col
        self.time_col = time_col
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length
        
        # Sort by user and time
        self.df = transactions_df.sort_values([user_col, time_col]).reset_index(drop=True)
        
        if feature_cols is None:
            # Use all columns except identifiers and target
            exclude_cols = [user_col, time_col, target_col, 'transaction_id']
            self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        else:
            self.feature_cols = feature_cols
        
        # Build sequences
        self.sequences = []
        self.targets = []
        self.user_ids = []
        
        self._build_sequences()
    
    def _build_sequences(self):
        """Build sequences for each user."""
        current_user = None
        user_transactions = []
        
        for idx, row in self.df.iterrows():
            user = row[self.user_col]
            
            if user != current_user:
                # New user, reset sequence
                if len(user_transactions) >= self.min_sequence_length:
                    self._add_user_sequences(user_transactions)
                user_transactions = []
                current_user = user
            
            user_transactions.append(row)
        
        # Add last user
        if len(user_transactions) >= self.min_sequence_length:
            self._add_user_sequences(user_transactions)
    
    def _add_user_sequences(self, user_transactions):
        """Add sequences for a single user."""
        n_transactions = len(user_transactions)
        
        for i in range(self.sequence_length, n_transactions):
            # Get sequence of previous transactions
            seq_start = i - self.sequence_length
            seq_end = i
            
            sequence = user_transactions[seq_start:seq_end]
            
            # Extract features
            seq_features = []
            for trans in sequence:
                features = [trans[col] for col in self.feature_cols]
                seq_features.append(features)
            
            # Target is the next transaction's fraud label
            target = user_transactions[i][self.target_col]
            
            self.sequences.append(np.array(seq_features))
            self.targets.append(target)
            self.user_ids.append(user_transactions[i][self.user_col])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.targets[idx]])[0]
        )


class LSTMFraudDetector(nn.Module):
    """
    LSTM model for sequential fraud detection.
    
    Architecture:
    - LSTM layers to capture temporal patterns
    - Attention mechanism for important transactions
    - Dense layers for final classification
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True, use_attention: bool = True):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Number of input features per transaction
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
        """
        super(LSTMFraudDetector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * (2 if bidirectional else 1),
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Calculate LSTM output dimension
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Dense layers for classification
        self.dense1 = nn.Linear(lstm_out_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout)
        
        self.dense2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout)
        
        self.output = nn.Linear(32, 2)
        
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, 2)
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.use_attention:
            # Self-attention over sequence
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            # Use attention-weighted average
            lstm_out = attn_out.mean(dim=1)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate last forward and backward hidden states
                hidden_forward = hidden[-2, :, :]  # Last forward layer
                hidden_backward = hidden[-1, :, :]  # Last backward layer
                lstm_out = torch.cat((hidden_forward, hidden_backward), dim=1)
            else:
                lstm_out = hidden[-1, :, :]  # Last hidden state
        
        # Dense layers
        x = self.dense1(lstm_out)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        output = self.output(x)
        
        return output


class FraudLSTMModel:
    """
    LSTM-based model for sequential fraud detection.
    
    Uses transaction sequences to detect fraud patterns over time.
    """
    
    def __init__(self, input_dim: int, config: Optional[Dict] = None,
                 device: Optional[str] = None, random_state: int = 42):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Number of input features per transaction
            config: Model configuration
            device: Device to use
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
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'use_attention': True,
            'sequence_length': 10,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 64,
            'epochs': 50,
            'early_stopping_patience': 10,
            'scheduler_factor': 0.5,
            'scheduler_patience': 5
        }
        
        if config:
            self.default_config.update(config)
        
        self.config = self.default_config
        
        # Build model
        self.model = LSTMFraudDetector(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            bidirectional=self.config['bidirectional'],
            use_attention=self.config['use_attention']
        ).to(self.device)
        
        self.sequence_length = self.config['sequence_length']
        self.feature_cols = None
        self.threshold = 0.5
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        logger.info(f"LSTM model initialized with {self.count_parameters():,} parameters")
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _create_dataloaders(self, X: pd.DataFrame, y: pd.Series,
                           batch_size: Optional[int] = None,
                           shuffle: bool = True) -> DataLoader:
        """
        Create DataLoader for sequence data.
        
        Args:
            X: Features DataFrame (must contain user_id and time columns)
            y: Labels Series
            batch_size: Batch size
            shuffle: Whether to shuffle
        
        Returns:
            DataLoader for sequences
        """
        if batch_size is None:
            batch_size = self.config['batch_size']
        
        # Create dataset
        dataset = TransactionSequenceDataset(
            transactions_df=pd.concat([X, y.to_frame('is_fraud')], axis=1),
            user_col='customer_id',  # Assume this column exists
            time_col='transaction_time',  # Assume this column exists
            feature_cols=self.feature_cols,
            target_col='is_fraud',
            sequence_length=self.sequence_length
        )
        
        # Store feature columns if not already set
        if self.feature_cols is None:
            self.feature_cols = dataset.feature_cols
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Set to >0 for faster loading
        )
        
        return dataloader
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              verbose: bool = True) -> 'FraudLSTMModel':
        """
        Train LSTM model on transaction sequences.
        
        Args:
            X_train: Training features (must include user_id and time)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            verbose: Whether to print progress
        
        Returns:
            Trained model
        """
        logger.info("Starting LSTM training...")
        
        # Create data loaders
        train_loader = self._create_dataloaders(X_train, y_train, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloaders(X_val, y_val, shuffle=False)
            use_validation = True
        else:
            val_loader = None
            use_validation = False
        
        # Calculate class weights
        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        class_weights = torch.tensor([1.0, pos_weight]).to(self.device)
        
        # Loss function
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        optimizer = optim.AdamW(
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
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
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
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
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
                val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
                
                self.training_history['val_loss'].append(avg_val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                self.training_history['val_f1'].append(val_f1)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if verbose and (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{self.config['epochs']}] - "
                        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                        f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                        f"Val F1: {val_f1:.4f}"
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
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probabilities for sequences.
        
        Args:
            X: Features DataFrame (must include user_id and time)
        
        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        self.model.eval()
        
        # Create dataloader
        dataloader = self._create_dataloaders(X, pd.Series([0] * len(X)), shuffle=False)
        
        all_probs = []
        
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.vstack(all_probs)
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary labels."""
        if threshold is None:
            threshold = self.threshold
        
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
    
    def find_optimal_threshold(self, X_val: pd.DataFrame,
                                y_val: pd.Series, metric: str = 'f1') -> float:
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
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 threshold: Optional[float] = None) -> Dict[str, float]:
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
            'feature_cols': self.feature_cols,
            'threshold': self.threshold,
            'training_history': self.training_history,
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'random_state': self.random_state,
            'saved_at': datetime.now().isoformat()
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'FraudLSTMModel':
        """Load model from disk."""
        model_data = torch.load(filepath, map_location=self.device)
        
        self.config = model_data['config']
        self.feature_cols = model_data['feature_cols']
        self.threshold = model_data['threshold']
        self.training_history = model_data['training_history']
        self.input_dim = model_data['input_dim']
        self.sequence_length = model_data['sequence_length']
        self.random_state = model_data['random_state']
        
        # Rebuild model
        self.model = LSTMFraudDetector(
            input_dim=self.input_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            bidirectional=self.config['bidirectional'],
            use_attention=self.config['use_attention']
        ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        return self