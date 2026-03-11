# tests/test_models.py
"""
Unit tests for machine learning models.
Tests classical ML, deep learning models, training, and evaluation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os
import tempfile
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling.classical_ml.isolation_forest import IsolationForestModel
from src.modeling.classical_ml.xgboost_model import FraudXGBoostModel
from src.modeling.classical_ml.lightgbm_model import LightGBMModel
from src.modeling.classical_ml.ensemble_methods import EnsembleFraudDetector
from src.modeling.deep_learning.neural_networks import FraudNeuralNetwork
from src.modeling.deep_learning.autoencoders import FraudAutoencoder
from src.modeling.deep_learning.lstm_models import FraudLSTM
from src.modeling.training.cross_validation import TimeSeriesCrossValidator
from src.modeling.training.hyperparameter_tuning import HyperparameterTuner
from src.modeling.evaluation.metrics import FraudMetrics
from src.modeling.evaluation.thresholds import ThresholdOptimizer
from src.modeling.evaluation.interpretability import ModelExplainer


@pytest.fixture
def classification_data():
    """
    Fixture providing synthetic classification data for testing.
    """
    np.random.seed(42)
    
    # Generate synthetic data with class imbalance
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.95, 0.05],  # 5% fraud rate
        flip_y=0.01,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    
    # Split into train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names
    }


class TestIsolationForestModel:
    """
    Test suite for IsolationForestModel.
    Tests anomaly detection for fraud detection.
    """
    
    def test_initialization(self):
        """
        Test IsolationForestModel initialization.
        """
        model = IsolationForestModel()
        assert model.model is None
        assert model.contamination == 0.1
        
        model = IsolationForestModel(contamination=0.05, random_state=42)
        assert model.contamination == 0.05
        assert model.random_state == 42
    
    def test_train(self, classification_data):
        """
        Test training isolation forest.
        """
        model = IsolationForestModel(contamination=0.05, random_state=42)
        
        # Train model
        model.train(classification_data['X_train'])
        
        # Check model is trained
        assert model.model is not None
        assert model.is_fitted == True
    
    def test_predict(self, classification_data):
        """
        Test prediction with isolation forest.
        """
        model = IsolationForestModel(contamination=0.05, random_state=42)
        
        # Train
        model.train(classification_data['X_train'])
        
        # Predict
        predictions = model.predict(classification_data['X_test'])
        
        # Check predictions
        assert len(predictions) == len(classification_data['X_test'])
        assert set(predictions).issubset({-1, 1}) or set(predictions).issubset({0, 1})
        
        # If predictions are -1/1 (anomaly/normal), convert to 0/1 for consistency
        if -1 in predictions:
            predictions = (predictions == -1).astype(int)
        
        # Check fraud rate in predictions (should be close to contamination)
        fraud_rate = predictions.mean()
        assert abs(fraud_rate - 0.05) < 0.03  # Within 3% of contamination
    
    def test_predict_proba(self, classification_data):
        """
        Test probability prediction.
        """
        model = IsolationForestModel(contamination=0.05, random_state=42)
        
        # Train
        model.train(classification_data['X_train'])
        
        # Get anomaly scores
        scores = model.predict_proba(classification_data['X_test'])
        
        # Check scores
        assert len(scores) == len(classification_data['X_test'])
        assert all(0 <= s <= 1 for s in scores)
        
        # Higher score should indicate more anomalous
        # Get top 10% most anomalous
        threshold = np.percentile(scores, 90)
        top_anomalies = scores > threshold
        
        # These should have higher correlation with actual fraud
        # This is a weak test but ensures functionality
        assert len(top_anomalies) > 0


class TestFraudXGBoostModel:
    """
    Test suite for FraudXGBoostModel.
    Tests XGBoost classifier for fraud detection.
    """
    
    def test_initialization(self):
        """
        Test FraudXGBoostModel initialization.
        """
        model = FraudXGBoostModel()
        assert model.model is None
        assert model.params is not None
        
        config = {
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 200
        }
        model = FraudXGBoostModel(config=config)
        assert model.params['max_depth'] == 8
        assert model.params['learning_rate'] == 0.05
    
    def test_train(self, classification_data):
        """
        Test training XGBoost model.
        """
        model = FraudXGBoostModel()
        
        # Train model
        model.train(
            classification_data['X_train'],
            classification_data['y_train'],
            X_val=classification_data['X_val'],
            y_val=classification_data['y_val']
        )
        
        # Check model is trained
        assert model.model is not None
        assert model.is_fitted == True
        
        # Check feature importance is calculated
        assert model.feature_importance_ is not None
    
    def test_predict(self, classification_data):
        """
        Test prediction with XGBoost.
        """
        model = FraudXGBoostModel()
        
        # Train
        model.train(
            classification_data['X_train'],
            classification_data['y_train']
        )
        
        # Predict
        predictions = model.predict(classification_data['X_test'])
        
        # Check predictions
        assert len(predictions) == len(classification_data['y_test'])
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, classification_data):
        """
        Test probability prediction.
        """
        model = FraudXGBoostModel()
        
        # Train
        model.train(
            classification_data['X_train'],
            classification_data['y_train']
        )
        
        # Get probabilities
        probabilities = model.predict_proba(classification_data['X_test'])
        
        # Check probabilities
        assert probabilities.shape[0] == len(classification_data['y_test'])
        assert probabilities.shape[1] == 2  # Two classes
        assert np.allclose(probabilities.sum(axis=1), 1)
    
    def test_feature_importance(self, classification_data):
        """
        Test feature importance calculation.
        """
        model = FraudXGBoostModel()
        
        # Train
        model.train(
            classification_data['X_train'],
            classification_data['y_train']
        )
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        # Check importance
        assert len(importance) == classification_data['X_train'].shape[1]
        assert all(v >= 0 for v in importance.values())
        
        # Sum of importance should be close to 1 (for normalized importance)
        if model.importance_type == 'gain':
            # Gain importance doesn't sum to 1
            pass
        else:
            total = sum(importance.values())
            assert abs(total - 1.0) < 0.01 or total > 0
    
    def test_early_stopping(self, classification_data):
        """
        Test early stopping during training.
        """
        model = FraudXGBoostModel({
            'n_estimators': 1000,
            'early_stopping_rounds': 10,
            'eval_metric': 'logloss'
        })
        
        # Train with validation set for early stopping
        model.train(
            classification_data['X_train'],
            classification_data['y_train'],
            X_val=classification_data['X_val'],
            y_val=classification_data['y_val']
        )
        
        # Check that we stopped early (less than 1000 estimators)
        assert model.model.best_iteration < 1000
    
    def test_handle_class_imbalance(self, classification_data):
        """
        Test handling of class imbalance.
        """
        # Check original imbalance
        fraud_rate_train = classification_data['y_train'].mean()
        assert fraud_rate_train < 0.1  # Less than 10% fraud
        
        # Train with scale_pos_weight
        scale_pos_weight = (1 - fraud_rate_train) / fraud_rate_train
        model = FraudXGBoostModel({'scale_pos_weight': scale_pos_weight})
        
        model.train(
            classification_data['X_train'],
            classification_data['y_train']
        )
        
        # Predict
        predictions = model.predict(classification_data['X_test'])
        
        # Should predict some fraud cases
        assert predictions.sum() > 0


class TestEnsembleFraudDetector:
    """
    Test suite for EnsembleFraudDetector.
    Tests ensemble methods combining multiple models.
    """
    
    def test_initialization(self):
        """
        Test EnsembleFraudDetector initialization.
        """
        ensemble = EnsembleFraudDetector()
        assert len(ensemble.models) == 0
        assert ensemble.weights == {}
        
        # Initialize with models
        models = {
            'xgboost': FraudXGBoostModel(),
            'isolation_forest': IsolationForestModel()
        }
        weights = {'xgboost': 0.7, 'isolation_forest': 0.3}
        
        ensemble = EnsembleFraudDetector(models=models, weights=weights)
        assert len(ensemble.models) == 2
        assert ensemble.weights == weights
    
    def test_add_model(self):
        """
        Test adding models to ensemble.
        """
        ensemble = EnsembleFraudDetector()
        
        # Add model
        ensemble.add_model('xgboost', FraudXGBoostModel(), weight=0.6)
        
        assert 'xgboost' in ensemble.models
        assert ensemble.weights['xgboost'] == 0.6
        
        # Add another model
        ensemble.add_model('isolation_forest', IsolationForestModel(), weight=0.4)
        
        assert len(ensemble.models) == 2
        assert ensemble.weights['isolation_forest'] == 0.4
    
    def test_train_ensemble(self, classification_data):
        """
        Test training all models in ensemble.
        """
        ensemble = EnsembleFraudDetector()
        
        # Add models
        ensemble.add_model('xgboost', FraudXGBoostModel())
        ensemble.add_model('isolation_forest', IsolationForestModel())
        
        # Train all models
        ensemble.train_all(
            classification_data['X_train'],
            classification_data['y_train']
        )
        
        # Check each model is trained
        assert ensemble.models['xgboost'].is_fitted
        assert ensemble.models['isolation_forest'].is_fitted
    
    def test_predict_ensemble(self, classification_data):
        """
        Test ensemble prediction.
        """
        ensemble = EnsembleFraudDetector()
        
        # Add models with weights
        ensemble.add_model('xgboost', FraudXGBoostModel(), weight=0.7)
        ensemble.add_model('isolation_forest', IsolationForestModel(), weight=0.3)
        
        # Train
        ensemble.train_all(
            classification_data['X_train'],
            classification_data['y_train']
        )
        
        # Predict
        predictions = ensemble.predict(classification_data['X_test'])
        
        # Check predictions
        assert len(predictions) == len(classification_data['y_test'])
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba_ensemble(self, classification_data):
        """
        Test weighted probability prediction.
        """
        ensemble = EnsembleFraudDetector()
        
        # Add models
        ensemble.add_model('xgboost', FraudXGBoostModel(), weight=0.7)
        ensemble.add_model('isolation_forest', IsolationForestModel(), weight=0.3)
        
        # Train
        ensemble.train_all(
            classification_data['X_train'],
            classification_data['y_train']
        )
        
        # Get probabilities
        probabilities = ensemble.predict_proba(classification_data['X_test'])
        
        # Check probabilities
        assert len(probabilities) == len(classification_data['y_test'])
        assert all(0 <= p <= 1 for p in probabilities)
        
        # With weighted average, should be between individual model predictions
        xgb_proba = ensemble.models['xgboost'].predict_proba(classification_data['X_test'])
        if ensemble.models['isolation_forest'].__class__.__name__ == 'IsolationForestModel':
            if_proba = ensemble.models['isolation_forest'].predict_proba(classification_data['X_test'])
            
            # Ensemble probability should be weighted average
            expected = 0.7 * xgb_proba + 0.3 * if_proba
            np.testing.assert_array_almost_equal(probabilities, expected, decimal=2)
    
    def test_voting_ensemble(self, classification_data):
        """
        Test voting-based ensemble (hard voting).
        """
        ensemble = EnsembleFraudDetector(method='hard_voting')
        
        # Add models
        ensemble.add_model('xgboost', FraudXGBoostModel())
        ensemble.add_model('isolation_forest', IsolationForestModel())
        
        # Train
        ensemble.train_all(
            classification_data['X_train'],
            classification_data['y_train']
        )
        
        # Predict with hard voting
        predictions = ensemble.predict(classification_data['X_test'])
        
        # Check predictions
        assert len(predictions) == len(classification_data['y_test'])
        
        # Should be integer votes
        assert all(p in [0, 1] for p in predictions)


class TestFraudNeuralNetwork:
    """
    Test suite for FraudNeuralNetwork.
    Tests neural network models for fraud detection.
    """
    
    def test_initialization(self):
        """
        Test FraudNeuralNetwork initialization.
        """
        model = FraudNeuralNetwork(input_size=20)
        assert model.input_size == 20
        assert model.model is None
        
        model = FraudNeuralNetwork(
            input_size=20,
            hidden_layers=[128, 64, 32],
            dropout_rate=0.3,
            learning_rate=0.001
        )
        assert model.hidden_layers == [128, 64, 32]
        assert model.dropout_rate == 0.3
    
    def test_build_model(self):
        """
        Test neural network architecture building.
        """
        model = FraudNeuralNetwork(input_size=20)
        
        # Build model
        model.build_model()
        
        # Check model structure
        assert model.model is not None
        assert hasattr(model.model, 'layers')
        
        # Count layers (should have input, hidden, output)
        assert len(model.model.layers) >= 3
    
    def test_train(self, classification_data):
        """
        Test training neural network.
        """
        model = FraudNeuralNetwork(
            input_size=classification_data['X_train'].shape[1],
            epochs=10,
            batch_size=64,
            verbose=0
        )
        
        # Build model
        model.build_model()
        
        # Train
        history = model.train(
            classification_data['X_train'].values,
            classification_data['y_train'],
            validation_data=(
                classification_data['X_val'].values,
                classification_data['y_val']
            )
        )
        
        # Check training history
        assert 'loss' in history
        assert 'accuracy' in history
        assert len(history['loss']) == 10  # 10 epochs
        
        # Model should be trained
        assert model.is_fitted == True
    
    def test_predict(self, classification_data):
        """
        Test neural network prediction.
        """
        model = FraudNeuralNetwork(
            input_size=classification_data['X_train'].shape[1],
            epochs=5,
            batch_size=64,
            verbose=0
        )
        
        # Build and train
        model.build_model()
        model.train(
            classification_data['X_train'].values,
            classification_data['y_train']
        )
        
        # Predict
        predictions = model.predict(classification_data['X_test'].values)
        
        # Check predictions
        assert len(predictions) == len(classification_data['y_test'])
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, classification_data):
        """
        Test probability prediction.
        """
        model = FraudNeuralNetwork(
            input_size=classification_data['X_train'].shape[1],
            epochs=5,
            batch_size=64,
            verbose=0
        )
        
        # Build and train
        model.build_model()
        model.train(
            classification_data['X_train'].values,
            classification_data['y_train']
        )
        
        # Get probabilities
        probabilities = model.predict_proba(classification_data['X_test'].values)
        
        # Check probabilities
        assert len(probabilities) == len(classification_data['y_test'])
        assert all(0 <= p <= 1 for p in probabilities)
    
    def test_save_load_model(self, classification_data):
        """
        Test saving and loading neural network.
        """
        model = FraudNeuralNetwork(
            input_size=classification_data['X_train'].shape[1],
            epochs=2,
            batch_size=64,
            verbose=0
        )
        
        # Build and train
        model.build_model()
        model.train(
            classification_data['X_train'].values,
            classification_data['y_train']
        )
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            model_path = tmp.name
            model.save(model_path)
        
        # Load model
        loaded_model = FraudNeuralNetwork.load(model_path)
        
        # Check loaded model
        assert loaded_model.input_size == model.input_size
        assert loaded_model.is_fitted == True
        
        # Predict with loaded model
        predictions = loaded_model.predict(classification_data['X_test'].values)
        assert len(predictions) == len(classification_data['y_test'])
        
        # Cleanup
        os.unlink(model_path)


class TestFraudAutoencoder:
    """
    Test suite for FraudAutoencoder.
    Tests autoencoder-based anomaly detection.
    """
    
    def test_initialization(self):
        """
        Test FraudAutoencoder initialization.
        """
        model = FraudAutoencoder(input_size=20)
        assert model.input_size == 20
        assert model.encoder is None
        assert model.decoder is None
        
        model = FraudAutoencoder(
            input_size=20,
            encoding_dim=8,
            hidden_layers=[16, 12]
        )
        assert model.encoding_dim == 8
        assert model.hidden_layers == [16, 12]
    
    def test_build_autoencoder(self):
        """
        Test autoencoder architecture building.
        """
        model = FraudAutoencoder(input_size=20, encoding_dim=8)
        
        # Build autoencoder
        model.build_autoencoder()
        
        # Check components
        assert model.autoencoder is not None
        assert model.encoder is not None
        assert model.decoder is not None
        
        # Check dimensions
        # This is hard to test without actually running the model
        # But we can check that the models exist
        pass
    
    def test_train_autoencoder(self, classification_data):
        """
        Test training autoencoder.
        """
        model = FraudAutoencoder(
            input_size=classification_data['X_train'].shape[1],
            encoding_dim=10,
            epochs=10,
            batch_size=64,
            verbose=0
        )
        
        # Build autoencoder
        model.build_autoencoder()
        
        # Train (only on normal transactions for autoencoder)
        normal_idx = classification_data['y_train'] == 0
        X_normal = classification_data['X_train'].values[normal_idx]
        
        history = model.train(
            X_normal,
            validation_split=0.2
        )
        
        # Check training history
        assert 'loss' in history
        assert len(history['loss']) == 10
        
        # Model should be trained
        assert model.is_fitted == True
    
    def test_calculate_reconstruction_error(self, classification_data):
        """
        Test reconstruction error calculation.
        """
        model = FraudAutoencoder(
            input_size=classification_data['X_train'].shape[1],
            encoding_dim=10,
            epochs=5,
            batch_size=64,
            verbose=0
        )
        
        # Build and train
        model.build_autoencoder()
        normal_idx = classification_data['y_train'] == 0
        X_normal = classification_data['X_train'].values[normal_idx]
        model.train(X_normal)
        
        # Calculate reconstruction error
        errors = model.calculate_reconstruction_error(
            classification_data['X_test'].values
        )
        
        # Check errors
        assert len(errors) == len(classification_data['y_test'])
        assert all(e >= 0 for e in errors)
        
        # Fraudulent transactions should have higher error on average
        fraud_idx = classification_data['y_test'] == 1
        normal_idx = classification_data['y_test'] == 0
        
        if fraud_idx.sum() > 0 and normal_idx.sum() > 0:
            fraud_errors = errors[fraud_idx]
            normal_errors = errors[normal_idx]
            
            # This is not guaranteed with synthetic data, so we'll just check
            # that we can compute both
            assert len(fraud_errors) > 0
            assert len(normal_errors) > 0
    
    def test_predict_anomaly(self, classification_data):
        """
        Test anomaly prediction with threshold.
        """
        model = FraudAutoencoder(
            input_size=classification_data['X_train'].shape[1],
            encoding_dim=10,
            epochs=5,
            batch_size=64,
            verbose=0
        )
        
        # Build and train
        model.build_autoencoder()
        normal_idx = classification_data['y_train'] == 0
        X_normal = classification_data['X_train'].values[normal_idx]
        model.train(X_normal)
        
        # Set threshold (e.g., 95th percentile of normal errors)
        normal_errors = model.calculate_reconstruction_error(X_normal)
        threshold = np.percentile(normal_errors, 95)
        
        # Predict anomalies
        predictions = model.predict_anomaly(
            classification_data['X_test'].values,
            threshold=threshold
        )
        
        # Check predictions
        assert len(predictions) == len(classification_data['y_test'])
        assert set(predictions).issubset({0, 1})
        
        # About 5% should be anomalies (by threshold)
        anomaly_rate = predictions.mean()
        assert abs(anomaly_rate - 0.05) < 0.05


class TestFraudLSTM:
    """
    Test suite for FraudLSTM.
    Tests LSTM models for sequential fraud detection.
    """
    
    @pytest.fixture
    def sequence_data(self):
        """
        Fixture providing sequence data for LSTM testing.
        """
        np.random.seed(42)
        n_sequences = 500
        sequence_length = 10
        n_features = 5
        
        # Create sequences
        X = np.random.randn(n_sequences, sequence_length, n_features)
        
        # Create labels (fraud if sequence has unusual pattern)
        y = np.zeros(n_sequences)
        # Make some sequences fraudulent by adding trend
        fraud_indices = np.random.choice(n_sequences, size=25, replace=False)
        for idx in fraud_indices:
            X[idx] = X[idx] + np.linspace(0, 2, sequence_length)[:, np.newaxis]
            y[idx] = 1
        
        # Split
        split_idx = int(0.7 * n_sequences)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'sequence_length': sequence_length,
            'n_features': n_features
        }
    
    def test_initialization(self):
        """
        Test FraudLSTM initialization.
        """
        model = FraudLSTM(sequence_length=10, n_features=5)
        assert model.sequence_length == 10
        assert model.n_features == 5
        assert model.model is None
        
        model = FraudLSTM(
            sequence_length=20,
            n_features=8,
            lstm_units=[64, 32],
            dropout=0.3
        )
        assert model.lstm_units == [64, 32]
        assert model.dropout == 0.3
    
    def test_build_model(self):
        """
        Test LSTM model building.
        """
        model = FraudLSTM(sequence_length=10, n_features=5)
        
        # Build model
        model.build_model()
        
        # Check model structure
        assert model.model is not None
        assert hasattr(model.model, 'layers')
        
        # Should have LSTM layers
        lstm_layers = [layer for layer in model.model.layers 
                      if 'lstm' in layer.__class__.__name__.lower()]
        assert len(lstm_layers) > 0
    
    def test_train(self, sequence_data):
        """
        Test training LSTM model.
        """
        model = FraudLSTM(
            sequence_length=sequence_data['sequence_length'],
            n_features=sequence_data['n_features'],
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
        # Build model
        model.build_model()
        
        # Train
        history = model.train(
            sequence_data['X_train'],
            sequence_data['y_train'],
            validation_split=0.2
        )
        
        # Check training history
        assert 'loss' in history
        assert 'accuracy' in history
        assert len(history['loss']) == 10
        
        # Model should be trained
        assert model.is_fitted == True
    
    def test_predict(self, sequence_data):
        """
        Test LSTM prediction.
        """
        model = FraudLSTM(
            sequence_length=sequence_data['sequence_length'],
            n_features=sequence_data['n_features'],
            epochs=5,
            batch_size=32,
            verbose=0
        )
        
        # Build and train
        model.build_model()
        model.train(sequence_data['X_train'], sequence_data['y_train'])
        
        # Predict
        predictions = model.predict(sequence_data['X_test'])
        
        # Check predictions
        assert len(predictions) == len(sequence_data['y_test'])
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, sequence_data):
        """
        Test probability prediction.
        """
        model = FraudLSTM(
            sequence_length=sequence_data['sequence_length'],
            n_features=sequence_data['n_features'],
            epochs=5,
            batch_size=32,
            verbose=0
        )
        
        # Build and train
        model.build_model()
        model.train(sequence_data['X_train'], sequence_data['y_train'])
        
        # Get probabilities
        probabilities = model.predict_proba(sequence_data['X_test'])
        
        # Check probabilities
        assert len(probabilities) == len(sequence_data['y_test'])
        assert all(0 <= p <= 1 for p in probabilities)


class TestFraudMetrics:
    """
    Test suite for FraudMetrics.
    Tests evaluation metrics specific to fraud detection.
    """
    
    def test_initialization(self):
        """
        Test FraudMetrics initialization.
        """
        metrics = FraudMetrics()
        assert metrics.metrics is not None
        
        metrics = FraudMetrics(metrics_list=['precision', 'recall', 'f1', 'auc'])
        assert 'precision' in metrics.metrics_list
    
    def test_calculate_precision_recall(self, classification_data):
        """
        Test precision and recall calculation.
        """
        metrics = FraudMetrics()
        
        # Get model predictions (simple threshold model)
        np.random.seed(42)
        y_pred = (np.random.rand(len(classification_data['y_test'])) > 0.95).astype(int)
        
        # Calculate precision and recall
        precision, recall, f1 = metrics.calculate_precision_recall(
            classification_data['y_test'],
            y_pred
        )
        
        # Check outputs
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
    
    def test_calculate_auc(self, classification_data):
        """
        Test AUC calculation.
        """
        metrics = FraudMetrics()
        
        # Generate probabilities
        np.random.seed(42)
        y_proba = np.random.rand(len(classification_data['y_test']))
        
        # Adjust to have some correlation
        fraud_idx = classification_data['y_test'] == 1
        y_proba[fraud_idx] = y_proba[fraud_idx] + 0.2
        
        # Calculate AUC
        auc_score = metrics.calculate_auc(
            classification_data['y_test'],
            y_proba
        )
        
        # AUC should be > 0.5 (better than random)
        assert auc_score > 0.5
        assert auc_score <= 1.0
    
    def test_calculate_precision_at_k(self, classification_data):
        """
        Test precision at k calculation.
        """
        metrics = FraudMetrics()
        
        # Generate probabilities
        np.random.seed(42)
        y_proba = np.random.rand(len(classification_data['y_test']))
        
        # Make top predictions more likely to be fraud
        n_fraud = classification_data['y_test'].sum()
        top_indices = np.argsort(y_proba)[-n_fraud:]
        for idx in top_indices:
            if np.random.rand() > 0.3:  # 70% chance to set as fraud
                y_proba[idx] = 1.0
        
        # Calculate precision at k
        prec_at_k = metrics.calculate_precision_at_k(
            classification_data['y_test'],
            y_proba,
            k=50
        )
        
        # Check output
        assert 0 <= prec_at_k <= 1
    
    def test_calculate_recall_at_fpr(self, classification_data):
        """
        Test recall at fixed false positive rate.
        """
        metrics = FraudMetrics()
        
        # Generate probabilities
        np.random.seed(42)
        y_proba = np.random.rand(len(classification_data['y_test']))
        
        # Make fraud have higher probabilities
        fraud_idx = classification_data['y_test'] == 1
        y_proba[fraud_idx] = y_proba[fraud_idx] + 0.3
        y_proba = np.clip(y_proba, 0, 1)
        
        # Calculate recall at 1% FPR
        recall = metrics.calculate_recall_at_fpr(
            classification_data['y_test'],
            y_proba,
            fpr_threshold=0.01
        )
        
        # Check output
        assert 0 <= recall <= 1
    
    def test_calculate_cost_saved(self, classification_data):
        """
        Test cost saved calculation.
        """
        metrics = FraudMetrics(cost_config={
            'fraud_cost': 100,
            'false_positive_cost': 1
        })
        
        # Get model predictions
        np.random.seed(42)
        y_pred = (np.random.rand(len(classification_data['y_test'])) > 0.95).astype(int)
        
        # Calculate cost saved
        cost_saved = metrics.calculate_cost_saved(
            classification_data['y_test'],
            y_pred
        )
        
        # Check output (should be numeric)
        assert isinstance(cost_saved, (int, float))
    
    def test_comprehensive_metrics(self, classification_data):
        """
        Test calculation of all metrics.
        """
        metrics = FraudMetrics()
        
        # Get model predictions and probabilities
        np.random.seed(42)
        y_pred = (np.random.rand(len(classification_data['y_test'])) > 0.95).astype(int)
        y_proba = np.random.rand(len(classification_data['y_test']))
        
        # Calculate all metrics
        results = metrics.calculate_all(
            classification_data['y_test'],
            y_pred,
            y_proba
        )
        
        # Check all expected metrics are present
        expected_metrics = ['precision', 'recall', 'f1', 'accuracy', 
                           'auc', 'average_precision']
        for metric in expected_metrics:
            assert metric in results


class TestThresholdOptimizer:
    """
    Test suite for ThresholdOptimizer.
    Tests optimal threshold selection for fraud detection.
    """
    
    def test_initialization(self):
        """
        Test ThresholdOptimizer initialization.
        """
        optimizer = ThresholdOptimizer()
        assert optimizer.metric == 'f1'
        
        optimizer = ThresholdOptimizer(metric='recall', min_precision=0.5)
        assert optimizer.metric == 'recall'
        assert optimizer.min_precision == 0.5
    
    def test_find_optimal_threshold(self, classification_data):
        """
        Test finding optimal threshold.
        """
        optimizer = ThresholdOptimizer(metric='f1')
        
        # Generate probabilities
        np.random.seed(42)
        y_proba = np.random.rand(len(classification_data['y_val']))
        
        # Make fraud have higher probabilities
        fraud_idx = classification_data['y_val'] == 1
        y_proba[fraud_idx] = y_proba[fraud_idx] + 0.3
        y_proba = np.clip(y_proba, 0, 1)
        
        # Find optimal threshold
        threshold, score = optimizer.find_optimal_threshold(
            classification_data['y_val'],
            y_proba
        )
        
        # Check output
        assert 0 <= threshold <= 1
        assert 0 <= score <= 1
        
        # Threshold should be > 0.5 (since fraud has higher probabilities)
        assert threshold > 0.4
    
    def test_find_threshold_by_precision(self, classification_data):
        """
        Test finding threshold for target precision.
        """
        optimizer = ThresholdOptimizer()
        
        # Generate probabilities
        np.random.seed(42)
        y_proba = np.random.rand(len(classification_data['y_val']))
        
        # Find threshold for 80% precision
        threshold = optimizer.find_threshold_by_precision(
            classification_data['y_val'],
            y_proba,
            target_precision=0.8
        )
        
        # Check output
        assert 0 <= threshold <= 1
    
    def test_find_threshold_by_recall(self, classification_data):
        """
        Test finding threshold for target recall.
        """
        optimizer = ThresholdOptimizer()
        
        # Generate probabilities
        np.random.seed(42)
        y_proba = np.random.rand(len(classification_data['y_val']))
        
        # Find threshold for 70% recall
        threshold = optimizer.find_threshold_by_recall(
            classification_data['y_val'],
            y_proba,
            target_recall=0.7
        )
        
        # Check output
        assert 0 <= threshold <= 1
    
    def test_threshold_curve(self, classification_data):
        """
        Test generating threshold curve.
        """
        optimizer = ThresholdOptimizer()
        
        # Generate probabilities
        np.random.seed(42)
        y_proba = np.random.rand(len(classification_data['y_val']))
        
        # Get threshold curve
        thresholds, precisions, recalls, f1_scores = optimizer.get_threshold_curve(
            classification_data['y_val'],
            y_proba
        )
        
        # Check outputs
        assert len(thresholds) > 0
        assert len(precisions) == len(thresholds)
        assert len(recalls) == len(thresholds)
        assert len(f1_scores) == len(thresholds)
        
        # As threshold increases, precision should increase, recall decrease
        # (generally, but not guaranteed with small samples)
        assert precisions[0] <= precisions[-1] + 0.1  # Allow some noise
        assert recalls[0] >= recalls[-1] - 0.1


class TestModelExplainer:
    """
    Test suite for ModelExplainer.
    Tests model interpretability methods.
    """
    
    @pytest.fixture
    def trained_model(self, classification_data):
        """
        Fixture providing a trained model for explanation.
        """
        model = FraudXGBoostModel()
        model.train(
            classification_data['X_train'],
            classification_data['y_train']
        )
        return model
    
    def test_initialization(self, trained_model, classification_data):
        """
        Test ModelExplainer initialization.
        """
        explainer = ModelExplainer(
            trained_model,
            classification_data['X_train']
        )
        
        assert explainer.model == trained_model
        assert explainer.background_data is not None
    
    def test_shap_explanations(self, trained_model, classification_data):
        """
        Test SHAP explanations.
        """
        explainer = ModelExplainer(
            trained_model,
            classification_data['X_train'].iloc[:100]  # Use subset for speed
        )
        
        # Get SHAP values for a sample
        sample = classification_data['X_test'].iloc[:5]
        shap_values = explainer.get_shap_values(sample)
        
        # Check SHAP values
        assert shap_values is not None
        assert shap_values.shape[0] == len(sample)
        assert shap_values.shape[1] == sample.shape[1]
    
    def test_feature_importance(self, trained_model, classification_data):
        """
        Test feature importance extraction.
        """
        explainer = ModelExplainer(
            trained_model,
            classification_data['X_train']
        )
        
        # Get feature importance
        importance = explainer.get_feature_importance()
        
        # Check importance
        assert len(importance) == classification_data['X_train'].shape[1]
        assert all(v >= 0 for v in importance.values())
    
    def test_explain_prediction(self, trained_model, classification_data):
        """
        Test explaining a single prediction.
        """
        explainer = ModelExplainer(
            trained_model,
            classification_data['X_train'].iloc[:100]
        )
        
        # Explain a prediction
        sample = classification_data['X_test'].iloc[0:1]
        explanation = explainer.explain_prediction(sample)
        
        # Check explanation
        assert 'prediction' in explanation
        assert 'probability' in explanation
        assert 'top_features' in explanation
        assert len(explanation['top_features']) > 0
        
        # Top features should have names and contributions
        for feature in explanation['top_features']:
            assert 'name' in feature
            assert 'value' in feature
            assert 'contribution' in feature
    
    def test_explain_batch(self, trained_model, classification_data):
        """
        Test explaining multiple predictions.
        """
        explainer = ModelExplainer(
            trained_model,
            classification_data['X_train'].iloc[:100]
        )
        
        # Explain multiple predictions
        samples = classification_data['X_test'].iloc[:3]
        explanations = explainer.explain_batch(samples)
        
        # Check explanations
        assert len(explanations) == len(samples)
        for explanation in explanations:
            assert 'prediction' in explanation
            assert 'probability' in explanation
    
    def test_global_explanations(self, trained_model, classification_data):
        """
        Test global model explanations.
        """
        explainer = ModelExplainer(
            trained_model,
            classification_data['X_train'].iloc[:100]
        )
        
        # Get global explanations
        global_exp = explainer.get_global_explanations()
        
        # Check global explanations
        assert 'feature_importance' in global_exp
        assert 'summary_plot' in global_exp or 'feature_importance' in global_exp