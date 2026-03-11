"""
Anomaly Detection Module for Fraud Detection
=============================================
This module provides comprehensive anomaly detection tools for identifying
suspicious patterns and outliers in banking transaction data.

Key Features:
1. Statistical outlier detection (Z-score, IQR, MAD)
2. Isolation Forest for unsupervised anomaly detection
3. Density-based methods (LOF, DBSCAN)
4. Time series anomaly detection
5. Ensemble methods combining multiple detectors
6. Explainable anomaly detection with feature contributions

The module helps identify:
- Unusual transaction amounts
- Suspicious timing patterns
- Abnormal behavior sequences
- Novel fraud patterns
- Data quality issues
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import warnings
import logging

# Configure logging
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Main class for comprehensive anomaly detection.
    
    This class provides a unified interface for multiple anomaly
    detection algorithms and combines their results.
    
    Example:
        >>> detector = AnomalyDetector(df, features=['Amount', 'V1', 'V2'])
        >>> results = detector.detect_anomalies(methods=['isolation_forest', 'lof'])
        >>> anomalies = detector.get_anomaly_scores()
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        contamination: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            data: Input DataFrame
            features: List of features to use for anomaly detection
            contamination: Expected proportion of outliers
            random_state: Random seed for reproducibility
        """
        self.data = data.copy()
        self.contamination = contamination
        self.random_state = random_state
        
        # Select features
        if features is None:
            # Use all numeric columns
            self.features = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.features = [f for f in features if f in data.columns]
        
        # Prepare data
        self.X = self.data[self.features].fillna(0).values
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Store results
        self.anomaly_scores = {}
        self.anomaly_predictions = {}
        self.explanations = {}
        
        logger.info(f"Initialized anomaly detector with {len(self.features)} features")
    
    def detect_anomalies(
        self,
        methods: List[str] = ['isolation_forest', 'lof', 'elliptic_envelope'],
        combine_methods: bool = True
    ) -> Dict[str, Any]:
        """
        Detect anomalies using multiple methods.
        
        Args:
            methods: List of anomaly detection methods to use
            combine_methods: Whether to combine results from all methods
        
        Returns:
            Dictionary with detection results
        """
        results = {}
        
        # Run each method
        for method in methods:
            if method == 'isolation_forest':
                results[method] = self._isolation_forest_detection()
            elif method == 'lof':
                results[method] = self._lof_detection()
            elif method == 'elliptic_envelope':
                results[method] = self._elliptic_envelope_detection()
            elif method == 'dbscan':
                results[method] = self._dbscan_detection()
            elif method == 'statistical':
                results[method] = self._statistical_detection()
        
        # Combine results if requested
        if combine_methods and len(methods) > 1:
            results['ensemble'] = self._combine_detections(results)
        
        return results
    
    def _isolation_forest_detection(self) -> Dict[str, Any]:
        """
        Detect anomalies using Isolation Forest.
        
        Isolation Forest isolates anomalies by randomly partitioning the data.
        Anomalies are easier to isolate and require fewer partitions.
        """
        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            bootstrap=False
        )
        
        # Fit and predict
        predictions = iso_forest.fit_predict(self.X_scaled)
        scores = iso_forest.decision_function(self.X_scaled)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        anomaly_flags = (predictions == -1).astype(int)
        
        # Store results
        self.anomaly_scores['isolation_forest'] = scores
        self.anomaly_predictions['isolation_forest'] = anomaly_flags
        
        # Feature importance (for tree-based methods)
        if hasattr(iso_forest, 'feature_importances_'):
            feature_importance = dict(zip(self.features, iso_forest.feature_importances_))
        else:
            # Estimate importance by running with single features
            feature_importance = self._estimate_isolation_forest_importance()
        
        return {
            'predictions': anomaly_flags.tolist(),
            'scores': scores.tolist(),
            'n_anomalies': int(anomaly_flags.sum()),
            'anomaly_rate': float(anomaly_flags.mean()),
            'feature_importance': feature_importance
        }
    
    def _estimate_isolation_forest_importance(self) -> Dict[str, float]:
        """Estimate feature importance for Isolation Forest."""
        importance = {}
        
        for i, feature in enumerate(self.features):
            # Use only this feature
            X_single = self.X_scaled[:, i:i+1]
            
            iso_forest = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state
            )
            iso_forest.fit(X_single)
            score = iso_forest.decision_function(X_single).mean()
            
            importance[feature] = float(abs(score))
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    def _lof_detection(self) -> Dict[str, Any]:
        """
        Detect anomalies using Local Outlier Factor.
        
        LOF measures the local density deviation of a point compared to its neighbors.
        Points with substantially lower density than neighbors are considered outliers.
        """
        # Train LOF
        lof = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=20,
            metric='euclidean'
        )
        
        # Fit and predict
        predictions = lof.fit_predict(self.X_scaled)
        
        # Negative LOF scores (more negative = more anomalous)
        scores = -lof.negative_outlier_factor_
        
        # Convert to binary
        anomaly_flags = (predictions == -1).astype(int)
        
        # Store results
        self.anomaly_scores['lof'] = scores
        self.anomaly_predictions['lof'] = anomaly_flags
        
        return {
            'predictions': anomaly_flags.tolist(),
            'scores': scores.tolist(),
            'n_anomalies': int(anomaly_flags.sum()),
            'anomaly_rate': float(anomaly_flags.mean())
        }
    
    def _elliptic_envelope_detection(self) -> Dict[str, Any]:
        """
        Detect anomalies using Elliptic Envelope.
        
        Assumes data is Gaussian and fits an ellipse to the central points.
        Points far from the ellipse are considered anomalies.
        """
        # Train Elliptic Envelope
        elliptic = EllipticEnvelope(
            contamination=self.contamination,
            random_state=self.random_state,
            support_fraction=0.7
        )
        
        try:
            # Fit and predict
            predictions = elliptic.fit_predict(self.X_scaled)
            scores = elliptic.decision_function(self.X_scaled)
            
            # Convert to binary
            anomaly_flags = (predictions == -1).astype(int)
            
        except Exception as e:
            logger.warning(f"Elliptic Envelope failed: {str(e)}")
            predictions = np.ones(len(self.X_scaled))
            scores = np.zeros(len(self.X_scaled))
            anomaly_flags = np.zeros(len(self.X_scaled))
        
        # Store results
        self.anomaly_scores['elliptic_envelope'] = scores
        self.anomaly_predictions['elliptic_envelope'] = anomaly_flags
        
        return {
            'predictions': anomaly_flags.tolist(),
            'scores': scores.tolist(),
            'n_anomalies': int(anomaly_flags.sum()),
            'anomaly_rate': float(anomaly_flags.mean())
        }
    
    def _dbscan_detection(self) -> Dict[str, Any]:
        """
        Detect anomalies using DBSCAN clustering.
        
        DBSCAN groups points that are closely packed together.
        Points that don't belong to any cluster are considered anomalies.
        """
        # Determine eps (neighborhood distance)
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(self.X_scaled)
        distances, _ = neigh.kneighbors(self.X_scaled)
        eps = np.percentile(distances[:, 4], 90)  # 90th percentile of 5th neighbor distances
        
        # Train DBSCAN
        dbscan = DBSCAN(
            eps=eps,
            min_samples=5,
            metric='euclidean'
        )
        
        # Fit and predict
        clusters = dbscan.fit_predict(self.X_scaled)
        
        # Points with cluster label -1 are anomalies
        anomaly_flags = (clusters == -1).astype(int)
        
        # Calculate anomaly scores (distance to nearest cluster)
        scores = np.zeros(len(self.X_scaled))
        for i in range(len(self.X_scaled)):
            if anomaly_flags[i] == 1:
                # For anomalies, score is distance to nearest cluster point
                cluster_points = self.X_scaled[clusters != -1]
                if len(cluster_points) > 0:
                    scores[i] = np.min(cdist([self.X_scaled[i]], cluster_points))
        
        # Store results
        self.anomaly_scores['dbscan'] = scores
        self.anomaly_predictions['dbscan'] = anomaly_flags
        
        return {
            'predictions': anomaly_flags.tolist(),
            'scores': scores.tolist(),
            'n_anomalies': int(anomaly_flags.sum()),
            'anomaly_rate': float(anomaly_flags.mean()),
            'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0)
        }
    
    def _statistical_detection(self) -> Dict[str, Any]:
        """
        Detect anomalies using statistical methods.
        
        Combines multiple statistical outlier detection methods:
        - Z-score
        - Modified Z-score (using MAD)
        - IQR
        """
        results = {}
        
        for feature in self.features:
            values = self.data[feature].dropna()
            
            if len(values) < 10:
                continue
            
            feature_results = {}
            
            # 1. Z-score method
            z_scores = np.abs(stats.zscore(values))
            z_anomalies = (z_scores > 3).astype(int)
            feature_results['z_score'] = {
                'anomalies': z_anomalies.tolist(),
                'n_anomalies': int(z_anomalies.sum()),
                'threshold': 3
            }
            
            # 2. Modified Z-score (using MAD)
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            if mad > 0:
                modified_z_scores = 0.6745 * (values - median) / mad
                modified_z_anomalies = (np.abs(modified_z_scores) > 3.5).astype(int)
            else:
                modified_z_anomalies = np.zeros(len(values))
            
            feature_results['modified_z_score'] = {
                'anomalies': modified_z_anomalies.tolist(),
                'n_anomalies': int(modified_z_anomalies.sum()),
                'threshold': 3.5
            }
            
            # 3. IQR method
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_anomalies = ((values < lower_bound) | (values > upper_bound)).astype(int)
            
            feature_results['iqr'] = {
                'anomalies': iqr_anomalies.tolist(),
                'n_anomalies': int(iqr_anomalies.sum()),
                'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
            }
            
            results[feature] = feature_results
        
        return results
    
    def _combine_detections(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine results from multiple detection methods.
        
        Uses voting mechanism to create ensemble predictions.
        """
        # Collect all predictions
        all_predictions = []
        method_names = []
        
        for method, result in results.items():
            if 'predictions' in result:
                all_predictions.append(np.array(result['predictions']))
                method_names.append(method)
        
        if not all_predictions:
            return {}
        
        # Stack predictions
        pred_stack = np.vstack(all_predictions)
        
        # Voting (simple majority)
        ensemble_predictions = (pred_stack.sum(axis=0) > len(method_names) / 2).astype(int)
        
        # Weighted voting (based on method performance - higher weight for conservative methods)
        weights = {'isolation_forest': 1.0, 'lof': 0.8, 'elliptic_envelope': 0.7, 'dbscan': 0.6}
        weighted_sum = np.zeros(len(self.X))
        
        for i, method in enumerate(method_names):
            weight = weights.get(method, 0.5)
            weighted_sum += pred_stack[i] * weight
        
        weighted_predictions = (weighted_sum > np.mean(list(weights.values()))).astype(int)
        
        # Calculate agreement between methods
        agreement = pred_stack.mean(axis=0)  # Proportion of methods flagging as anomaly
        
        return {
            'ensemble_predictions': ensemble_predictions.tolist(),
            'weighted_predictions': weighted_predictions.tolist(),
            'method_agreement': agreement.tolist(),
            'n_anomalies_ensemble': int(ensemble_predictions.sum()),
            'n_anomalies_weighted': int(weighted_predictions.sum()),
            'method_names': method_names
        }
    
    def get_anomaly_scores(self, method: str = 'ensemble') -> pd.DataFrame:
        """
        Get anomaly scores for all data points.
        
        Args:
            method: Which method's scores to return
        
        Returns:
            DataFrame with anomaly scores
        """
        if method == 'ensemble' and 'ensemble' in self.anomaly_scores:
            scores = self.anomaly_scores['ensemble']
        elif method in self.anomaly_scores:
            scores = self.anomaly_scores[method]
        else:
            # Average of all available scores
            all_scores = np.array(list(self.anomaly_scores.values()))
            scores = all_scores.mean(axis=0)
        
        # Create DataFrame
        result_df = pd.DataFrame({
            'anomaly_score': scores,
            'is_anomaly': scores > np.percentile(scores, (1 - self.contamination) * 100)
        })
        
        return result_df
    
    def explain_anomalies(self, n_top_features: int = 5) -> pd.DataFrame:
        """
        Explain why points are flagged as anomalies.
        
        Args:
            n_top_features: Number of top contributing features to return
        
        Returns:
            DataFrame with anomaly explanations
        """
        explanations = []
        
        # Get anomaly predictions
        if 'ensemble' in self.anomaly_predictions:
            predictions = self.anomaly_predictions['ensemble']
        else:
            # Use first available method
            first_method = next(iter(self.anomaly_predictions))
            predictions = self.anomaly_predictions[first_method]
        
        anomaly_indices = np.where(predictions == 1)[0]
        
        for idx in anomaly_indices[:100]:  # Limit to first 100 anomalies
            point = self.X[idx]
            point_scaled = self.X_scaled[idx]
            
            # Calculate deviation from mean for each feature
            feature_means = self.X.mean(axis=0)
            feature_stds = self.X.std(axis=0)
            
            deviations = []
            for i, feature in enumerate(self.features):
                if feature_stds[i] > 0:
                    z_score = abs((point[i] - feature_means[i]) / feature_stds[i])
                    deviations.append((feature, z_score, point[i]))
            
            # Sort by deviation
            deviations.sort(key=lambda x: x[1], reverse=True)
            
            # Get top contributing features
            top_features = []
            for feature, z_score, value in deviations[:n_top_features]:
                top_features.append({
                    'feature': feature,
                    'z_score': float(z_score),
                    'value': float(value),
                    'mean': float(feature_means[i]),
                    'deviation': f"{z_score:.2f}σ {'above' if value > feature_means[i] else 'below'} normal"
                })
            
            explanations.append({
                'index': int(idx),
                'anomaly_score': float(self.anomaly_scores.get('ensemble', [0])[idx] 
                                      if 'ensemble' in self.anomaly_scores else 0),
                'top_features': top_features
            })
        
        return pd.DataFrame(explanations)


class OutlierAnalyzer:
    """
    Specialized class for outlier analysis.
    
    Focuses on statistical outlier detection and provides
    detailed analysis of outlier patterns.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def detect_univariate_outliers(self, method: str = 'iqr', threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect univariate outliers in each column.
        
        Args:
            method: Detection method ('iqr', 'zscore', 'mad')
            threshold: Threshold for outlier detection
        
        Returns:
            Dictionary with outlier detection results
        """
        results = {}
        
        for col in self.data.select_dtypes(include=[np.number]).columns:
            values = self.data[col].dropna()
            
            if len(values) < 10:
                continue
            
            if method == 'iqr':
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                outlier_indices = values.index[(values < lower_bound) | (values > upper_bound)].tolist()
                
                results[col] = {
                    'n_outliers': len(outliers),
                    'outlier_rate': len(outliers) / len(values),
                    'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)},
                    'outlier_indices': outlier_indices,
                    'outlier_values': outliers.tolist()
                }
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(values))
                outliers = values[z_scores > threshold]
                outlier_indices = values.index[z_scores > threshold].tolist()
                
                results[col] = {
                    'n_outliers': len(outliers),
                    'outlier_rate': len(outliers) / len(values),
                    'threshold': threshold,
                    'outlier_indices': outlier_indices,
                    'outlier_values': outliers.tolist()
                }
            
            elif method == 'mad':
                median = np.median(values)
                mad = np.median(np.abs(values - median))
                if mad > 0:
                    modified_z_scores = 0.6745 * (values - median) / mad
                    outliers = values[np.abs(modified_z_scores) > threshold]
                    outlier_indices = values.index[np.abs(modified_z_scores) > threshold].tolist()
                else:
                    outliers = pd.Series([])
                    outlier_indices = []
                
                results[col] = {
                    'n_outliers': len(outliers),
                    'outlier_rate': len(outliers) / len(values),
                    'threshold': threshold,
                    'outlier_indices': outlier_indices,
                    'outlier_values': outliers.tolist()
                }
        
        return results
    
    def detect_multivariate_outliers(self, n_components: int = 2) -> Dict[str, Any]:
        """
        Detect multivariate outliers using PCA-based method.
        
        Args:
            n_components: Number of principal components to use
        
        Returns:
            Dictionary with multivariate outlier detection results
        """
        # Select numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {}
        
        # Prepare data
        X = self.data[numeric_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, len(numeric_cols)))
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate Mahalanobis distance in PCA space
        cov = np.cov(X_pca.T)
        try:
            inv_cov = np.linalg.inv(cov)
        except:
            inv_cov = np.linalg.pinv(cov)
        
        mean = np.mean(X_pca, axis=0)
        
        mahalanobis_dist = []
        for point in X_pca:
            diff = point - mean
            dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
            mahalanobis_dist.append(dist)
        
        mahalanobis_dist = np.array(mahalanobis_dist)
        
        # Determine threshold (chi-square distribution)
        from scipy.stats import chi2
        threshold = chi2.ppf(0.975, df=n_components)
        
        # Identify outliers
        outliers = mahalanobis_dist > threshold
        outlier_indices = np.where(outliers)[0].tolist()
        
        return {
            'n_outliers': int(outliers.sum()),
            'outlier_rate': float(outliers.mean()),
            'threshold': float(threshold),
            'mahalanobis_distances': mahalanobis_dist.tolist(),
            'outlier_indices': outlier_indices,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
        }


class IsolationForestDetector:
    """
    Specialized Isolation Forest implementation with enhancements for fraud detection.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[int, float] = 'auto',
        random_state: int = 42
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.model = None
    
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit the model."""
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            bootstrap=False
        )
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        return self.model.predict(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        return self.model.decision_function(X)
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        For Isolation Forest, importance is estimated by measuring
        the impact of each feature on the anomaly scores.
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        # This is a simplified importance estimation
        # For more accurate importance, consider using permutation importance
        importance = {}
        
        # Get the trees
        estimators = self.model.estimators_
        
        # Count feature usage in splits
        feature_counts = defaultdict(int)
        total_splits = 0
        
        for tree in estimators:
            tree_ = tree.tree_
            features = tree_.feature
            features = features[features >= 0]  # Ignore leaf nodes
            for f in features:
                feature_counts[f] += 1
                total_splits += 1
        
        # Normalize
        for i, name in enumerate(feature_names):
            if total_splits > 0:
                importance[name] = feature_counts[i] / total_splits
            else:
                importance[name] = 0
        
        return importance


class StatisticalOutlierDetector:
    """
    Statistical methods for outlier detection.
    
    Provides various statistical tests and methods for identifying outliers.
    """
    
    def __init__(self, data: pd.Series):
        self.data = data
    
    def grubbs_test(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform Grubbs' test for outliers.
        
        Tests if the maximum value is an outlier.
        """
        n = len(self.data)
        mean = self.data.mean()
        std = self.data.std()
        
        if std == 0:
            return {'has_outliers': False, 'outliers': []}
        
        # Calculate G statistic
        max_val = self.data.max()
        g_calc = (max_val - mean) / std
        
        # Critical value
        from scipy.stats import t
        t_crit = t.ppf(1 - alpha / (2 * n), n - 2)
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        
        is_outlier = g_calc > g_crit
        
        return {
            'has_outliers': is_outlier,
            'outliers': [max_val] if is_outlier else [],
            'statistic': float(g_calc),
            'critical_value': float(g_crit),
            'p_value': float(1 - stats.norm.cdf(g_calc))
        }
    
    def dixon_test(self) -> Dict[str, Any]:
        """
        Perform Dixon's Q test for outliers.
        
        Suitable for small sample sizes (n < 30).
        """
        sorted_data = self.data.sort_values().values
        n = len(sorted_data)
        
        if n < 3 or n > 30:
            return {'has_outliers': False, 'error': 'Sample size not suitable for Dixon test'}
        
        # Calculate Q statistic
        if n <= 7:
            # Test smallest or largest
            gap = sorted_data[1] - sorted_data[0]
            range_ = sorted_data[-1] - sorted_data[0]
            q_calc = gap / range_ if range_ > 0 else 0
        elif n <= 10:
            # Test smallest or largest with more separation
            gap = sorted_data[1] - sorted_data[0]
            range_ = sorted_data[-2] - sorted_data[0]
            q_calc = gap / range_ if range_ > 0 else 0
        else:
            # Test smallest or largest with even more separation
            gap = sorted_data[2] - sorted_data[0]
            range_ = sorted_data[-2] - sorted_data[0]
            q_calc = gap / range_ if range_ > 0 else 0
        
        # Critical values (simplified)
        critical_values = {
            3: 0.941, 4: 0.765, 5: 0.642, 6: 0.560, 7: 0.507,
            8: 0.554, 9: 0.512, 10: 0.477, 11: 0.576, 12: 0.546,
            13: 0.521, 14: 0.546, 15: 0.525, 16: 0.507, 17: 0.490,
            18: 0.475, 19: 0.462, 20: 0.450, 21: 0.440, 22: 0.430,
            23: 0.421, 24: 0.413, 25: 0.406, 26: 0.399, 27: 0.393,
            28: 0.387, 29: 0.381, 30: 0.376
        }
        
        q_crit = critical_values.get(n, 0.5)
        is_outlier = q_calc > q_crit
        
        return {
            'has_outliers': is_outlier,
            'outliers': [sorted_data[0]] if is_outlier else [],
            'statistic': float(q_calc),
            'critical_value': q_crit
        }


class DensityBasedDetector:
    """
    Density-based anomaly detection methods.
    
    Uses local density estimation to identify anomalies.
    """
    
    def __init__(self, data: pd.DataFrame, features: List[str]):
        self.data = data
        self.features = features
        self.X = data[features].fillna(0).values
        
        # Standardize
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
    
    def detect_lof(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect anomalies using Local Outlier Factor.
        """
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric='euclidean'
        )
        
        predictions = lof.fit_predict(self.X_scaled)
        scores = -lof.negative_outlier_factor_
        
        anomaly_indices = np.where(predictions == -1)[0].tolist()
        
        return {
            'predictions': predictions.tolist(),
            'scores': scores.tolist(),
            'anomaly_indices': anomaly_indices,
            'n_anomalies': len(anomaly_indices)
        }
    
    def detect_dbscan(
        self,
        eps: Optional[float] = None,
        min_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Detect anomalies using DBSCAN.
        """
        if eps is None:
            # Estimate eps using k-distance graph
            from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors(n_neighbors=min_samples)
            neigh.fit(self.X_scaled)
            distances, _ = neigh.kneighbors(self.X_scaled)
            eps = np.percentile(distances[:, -1], 90)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(self.X_scaled)
        
        anomaly_indices = np.where(clusters == -1)[0].tolist()
        
        return {
            'clusters': clusters.tolist(),
            'anomaly_indices': anomaly_indices,
            'n_anomalies': len(anomaly_indices),
            'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
            'eps': float(eps)
        }
    
    def detect_knn(
        self,
        n_neighbors: int = 5,
        threshold_percentile: float = 95
    ) -> Dict[str, Any]:
        """
        Detect anomalies using k-nearest neighbor distances.
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Find k-nearest neighbors
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(self.X_scaled)
        distances, _ = neigh.kneighbors(self.X_scaled)
        
        # Use distance to kth neighbor as anomaly score
        k_distances = distances[:, -1]
        
        # Determine threshold
        threshold = np.percentile(k_distances, threshold_percentile)
        
        # Identify anomalies
        anomaly_indices = np.where(k_distances > threshold)[0].tolist()
        anomaly_scores = k_distances.tolist()
        
        return {
            'anomaly_indices': anomaly_indices,
            'anomaly_scores': anomaly_scores,
            'n_anomalies': len(anomaly_indices),
            'threshold': float(threshold),
            'k_distances': k_distances.tolist()
        }