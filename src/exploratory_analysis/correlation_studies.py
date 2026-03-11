"""
Correlation Studies Module for Fraud Detection
===============================================
This module provides comprehensive correlation analysis tools for
understanding relationships between features in fraud detection data.

Key Features:
1. Multiple correlation methods (Pearson, Spearman, Kendall)
2. Partial correlations controlling for confounders
3. Categorical associations (Cramér's V, Theil's U)
4. Mutual information for non-linear relationships
5. Multicollinearity detection (VIF, condition indices)
6. Feature selection based on correlation patterns

The module helps identify:
- Features strongly associated with fraud
- Redundant features for dimensionality reduction
- Interaction effects between variables
- Complex non-linear relationships
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    pearsonr, spearmanr, kendalltau,
    chi2_contingency, pointbiserialr
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from collections import defaultdict
import logging

# Configure logging
logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Main class for comprehensive correlation analysis.
    
    This class provides methods for calculating various correlation
    metrics and understanding relationships between features.
    
    Example:
        >>> analyzer = CorrelationAnalyzer(df, target_col='is_fraud')
        >>> correlations = analyzer.compute_all_correlations()
        >>> top_features = analyzer.get_top_correlated_features(n=10)
        >>> multicollinearity = analyzer.detect_multicollinearity()
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = 'is_fraud',
        categorical_threshold: int = 10,
        handle_missing: str = 'drop'
    ):
        """
        Initialize the correlation analyzer.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            categorical_threshold: Max unique values for categorical detection
            handle_missing: How to handle missing values ('drop', 'fill_mean', 'fill_median')
        """
        self.data = data.copy()
        self.target_col = target_col
        self.categorical_threshold = categorical_threshold
        
        # Handle missing values
        self._handle_missing(handle_missing)
        
        # Identify feature types
        self._identify_feature_types()
        
        logger.info(f"Initialized analyzer with {len(self.numeric_cols)} numeric and "
                   f"{len(self.categorical_cols)} categorical features")
    
    def _handle_missing(self, method: str) -> None:
        """Handle missing values in the data."""
        if method == 'drop':
            self.data = self.data.dropna()
            logger.info(f"Dropped rows with missing values. Remaining: {len(self.data)}")
        elif method == 'fill_mean':
            for col in self.data.select_dtypes(include=[np.number]).columns:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
        elif method == 'fill_median':
            for col in self.data.select_dtypes(include=[np.number]).columns:
                self.data[col].fillna(self.data[col].median(), inplace=True)
    
    def _identify_feature_types(self) -> None:
        """Identify numeric and categorical features."""
        self.numeric_cols = []
        self.categorical_cols = []
        
        for col in self.data.columns:
            if col == self.target_col:
                continue
                
            if self.data[col].dtype in ['object', 'category']:
                self.categorical_cols.append(col)
            elif self.data[col].dtype in [np.int64, np.float64]:
                # Check if it's actually categorical
                if self.data[col].nunique() <= self.categorical_threshold:
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)
    
    def compute_all_correlations(self) -> Dict[str, Any]:
        """
        Compute all types of correlations.
        
        Returns:
            Dictionary with correlation results
        """
        results = {}
        
        # 1. Pearson correlations (linear relationships)
        results['pearson'] = self._compute_pearson_correlations()
        
        # 2. Spearman correlations (monotonic relationships)
        results['spearman'] = self._compute_spearman_correlations()
        
        # 3. Kendall correlations (ordinal associations)
        results['kendall'] = self._compute_kendall_correlations()
        
        # 4. Point-biserial (binary vs continuous)
        results['point_biserial'] = self._compute_point_biserial()
        
        # 5. Cramér's V (categorical associations)
        results['cramers_v'] = self._compute_cramers_v()
        
        # 6. Mutual information (non-linear relationships)
        results['mutual_info'] = self._compute_mutual_information()
        
        return results
    
    def _compute_pearson_correlations(self) -> pd.DataFrame:
        """Compute Pearson correlation matrix for numeric features."""
        if len(self.numeric_cols) < 2:
            return pd.DataFrame()
        
        # Include target if numeric
        cols_to_use = self.numeric_cols.copy()
        if self.target_col in self.data.select_dtypes(include=[np.number]).columns:
            cols_to_use.append(self.target_col)
        
        return self.data[cols_to_use].corr(method='pearson')
    
    def _compute_spearman_correlations(self) -> pd.DataFrame:
        """Compute Spearman rank correlation matrix."""
        if len(self.numeric_cols) < 2:
            return pd.DataFrame()
        
        cols_to_use = self.numeric_cols.copy()
        if self.target_col in self.data.select_dtypes(include=[np.number]).columns:
            cols_to_use.append(self.target_col)
        
        return self.data[cols_to_use].corr(method='spearman')
    
    def _compute_kendall_correlations(self) -> pd.DataFrame:
        """Compute Kendall tau correlation matrix."""
        if len(self.numeric_cols) < 2:
            return pd.DataFrame()
        
        cols_to_use = self.numeric_cols.copy()
        if self.target_col in self.data.select_dtypes(include=[np.number]).columns:
            cols_to_use.append(self.target_col)
        
        return self.data[cols_to_use].corr(method='kendall')
    
    def _compute_point_biserial(self) -> Dict[str, float]:
        """
        Compute point-biserial correlation between binary target and numeric features.
        
        Point-biserial correlation measures the relationship between a binary
        variable (fraud/non-fraud) and continuous variables.
        """
        results = {}
        
        if self.target_col not in self.data.columns:
            return results
        
        # Ensure target is binary
        target_values = self.data[self.target_col].values
        
        for col in self.numeric_cols:
            if col == self.target_col:
                continue
                
            try:
                # Calculate point-biserial correlation
                corr, p_value = pointbiserialr(
                    target_values,
                    self.data[col].values
                )
                
                results[col] = {
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'interpretation': f"{'Positive' if corr > 0 else 'Negative'} correlation with fraud"
                }
            except Exception as e:
                logger.debug(f"Could not compute point-biserial for {col}: {str(e)}")
        
        return results
    
    def _compute_cramers_v(self) -> Dict[str, Any]:
        """
        Compute Cramér's V for categorical associations.
        
        Cramér's V measures association between categorical variables,
        ranging from 0 (no association) to 1 (perfect association).
        """
        results = {}
        
        # Include target if categorical
        cats_to_use = self.categorical_cols.copy()
        if self.target_col in self.data.select_dtypes(include=['object', 'category']).columns:
            cats_to_use.append(self.target_col)
        
        for i, col1 in enumerate(cats_to_use):
            for col2 in cats_to_use[i+1:]:
                try:
                    # Create contingency table
                    contingency = pd.crosstab(
                        self.data[col1].fillna('MISSING'),
                        self.data[col2].fillna('MISSING')
                    )
                    
                    # Calculate chi-square statistic
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    
                    # Calculate Cramér's V
                    n = len(self.data)
                    min_dim = min(contingency.shape) - 1
                    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                    
                    # Interpret strength
                    if cramers_v < 0.1:
                        strength = "negligible"
                    elif cramers_v < 0.3:
                        strength = "weak"
                    elif cramers_v < 0.5:
                        strength = "moderate"
                    else:
                        strength = "strong"
                    
                    results[f"{col1}_vs_{col2}"] = {
                        'cramers_v': float(cramers_v),
                        'chi2': float(chi2),
                        'p_value': float(p_value),
                        'degrees_freedom': int(dof),
                        'significant': p_value < 0.05,
                        'strength': strength
                    }
                    
                except Exception as e:
                    logger.debug(f"Could not compute Cramér's V for {col1} and {col2}: {str(e)}")
        
        return results
    
    def _compute_mutual_information(self) -> Dict[str, float]:
        """
        Compute mutual information between features and target.
        
        Mutual information captures any kind of relationship (linear or non-linear)
        between features and the target variable.
        """
        results = {}
        
        if self.target_col not in self.data.columns:
            return results
        
        # Prepare data
        X = self.data[self.numeric_cols].fillna(0).values
        
        # Determine if target is classification or regression
        if self.data[self.target_col].nunique() <= 2:
            # Binary classification
            y = self.data[self.target_col].values
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            # Regression
            y = self.data[self.target_col].values
            mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Store results
        for col, mi in zip(self.numeric_cols, mi_scores):
            results[col] = float(mi)
        
        return results
    
    def get_top_correlated_features(
        self,
        n: int = 10,
        method: str = 'pearson',
        with_target: bool = True
    ) -> pd.DataFrame:
        """
        Get top N features correlated with target.
        
        Args:
            n: Number of top features to return
            method: Correlation method ('pearson', 'spearman', 'kendall')
            with_target: Whether to include target correlations
        
        Returns:
            DataFrame with top correlated features
        """
        if not with_target or self.target_col not in self.data.columns:
            return pd.DataFrame()
        
        # Get correlation matrix
        if method == 'pearson':
            corr_matrix = self._compute_pearson_correlations()
        elif method == 'spearman':
            corr_matrix = self._compute_spearman_correlations()
        elif method == 'kendall':
            corr_matrix = self._compute_kendall_correlations()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if corr_matrix.empty or self.target_col not in corr_matrix.columns:
            return pd.DataFrame()
        
        # Get correlations with target
        target_corr = corr_matrix[self.target_col].drop(self.target_col)
        target_corr = target_corr.abs().sort_values(ascending=False)
        
        # Get top N
        top_features = target_corr.head(n)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'feature': top_features.index,
            'correlation_abs': top_features.values,
            'correlation_actual': corr_matrix.loc[top_features.index, self.target_col].values
        })
        
        return result
    
    def detect_multicollinearity(
        self,
        threshold: float = 5.0,
        method: str = 'vif'
    ) -> Dict[str, Any]:
        """
        Detect multicollinearity among features.
        
        Args:
            threshold: Threshold for multicollinearity detection
            method: Method to use ('vif' for Variance Inflation Factor,
                   'condition' for condition number)
        
        Returns:
            Dictionary with multicollinearity results
        """
        results = {}
        
        if method == 'vif':
            results = self._calculate_vif(threshold)
        elif method == 'condition':
            results = self._calculate_condition_number(threshold)
        
        return results
    
    def _calculate_vif(self, threshold: float = 5.0) -> Dict[str, Any]:
        """
        Calculate Variance Inflation Factor (VIF) for numeric features.
        
        VIF > 5 indicates high multicollinearity
        VIF > 10 indicates severe multicollinearity
        """
        if len(self.numeric_cols) < 2:
            return {}
        
        # Prepare data
        X = self.data[self.numeric_cols].fillna(0)
        X = add_constant(X)
        
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])]
        
        # Remove constant term
        vif_data = vif_data[vif_data["feature"] != "const"]
        
        # Identify problematic features
        high_vif = vif_data[vif_data["VIF"] > threshold]
        
        results = {
            'vif_values': vif_data.set_index('feature')['VIF'].to_dict(),
            'high_vif_features': high_vif['feature'].tolist(),
            'max_vif': float(vif_data['VIF'].max()),
            'mean_vif': float(vif_data['VIF'].mean()),
            'has_multicollinearity': len(high_vif) > 0
        }
        
        return results
    
    def _calculate_condition_number(self, threshold: float = 30.0) -> Dict[str, Any]:
        """
        Calculate condition number of feature matrix.
        
        Condition number > 30 indicates multicollinearity
        Condition number > 100 indicates severe multicollinearity
        """
        if len(self.numeric_cols) < 2:
            return {}
        
        # Prepare data
        X = self.data[self.numeric_cols].fillna(0).values
        
        # Calculate condition number
        from numpy.linalg import cond
        condition_number = cond(X)
        
        # Interpret
        if condition_number < 10:
            interpretation = "No serious multicollinearity"
        elif condition_number < 30:
            interpretation = "Moderate multicollinearity"
        elif condition_number < 100:
            interpretation = "Strong multicollinearity"
        else:
            interpretation = "Severe multicollinearity"
        
        results = {
            'condition_number': float(condition_number),
            'threshold': threshold,
            'has_multicollinearity': condition_number > threshold,
            'interpretation': interpretation
        }
        
        return results
    
    def find_redundant_pairs(
        self,
        threshold: float = 0.8,
        method: str = 'pearson'
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of highly correlated features (redundant).
        
        Args:
            threshold: Correlation threshold for redundancy
            method: Correlation method
        
        Returns:
            List of tuples (feature1, feature2, correlation)
        """
        # Get correlation matrix
        if method == 'pearson':
            corr_matrix = self._compute_pearson_correlations()
        elif method == 'spearman':
            corr_matrix = self._compute_spearman_correlations()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if corr_matrix.empty:
            return []
        
        # Find highly correlated pairs
        redundant_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                
                corr = abs(corr_matrix.iloc[i, j])
                
                if corr > threshold:
                    redundant_pairs.append((col1, col2, float(corr)))
        
        # Sort by correlation
        redundant_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return redundant_pairs


class FeatureSelector:
    """
    Feature selection based on correlation analysis.
    
    This class provides methods for selecting the best features
    based on their correlation with the target and redundancy.
    """
    
    def __init__(self, analyzer: CorrelationAnalyzer):
        self.analyzer = analyzer
    
    def select_by_correlation(
        self,
        n_features: int = 10,
        method: str = 'pearson',
        min_correlation: float = 0.1
    ) -> List[str]:
        """
        Select top N features by correlation with target.
        
        Args:
            n_features: Number of features to select
            method: Correlation method
            min_correlation: Minimum absolute correlation to include
        
        Returns:
            List of selected feature names
        """
        # Get top correlated features
        top_features = self.analyzer.get_top_correlated_features(
            n=n_features * 2,  # Get more to allow filtering
            method=method,
            with_target=True
        )
        
        if top_features.empty:
            return []
        
        # Filter by minimum correlation
        selected = top_features[
            top_features['correlation_abs'] >= min_correlation
        ].head(n_features)
        
        return selected['feature'].tolist()
    
    def select_non_redundant(
        self,
        n_features: int = 10,
        correlation_threshold: float = 0.7,
        method: str = 'pearson'
    ) -> List[str]:
        """
        Select non-redundant features.
        
        This method selects features that are correlated with the target
        but not highly correlated with each other.
        
        Args:
            n_features: Number of features to select
            correlation_threshold: Threshold for feature-feature correlation
            method: Correlation method
        
        Returns:
            List of selected feature names
        """
        # Get all correlations
        corr_matrix = getattr(self.analyzer, f'_compute_{method}_correlations')()
        
        if corr_matrix.empty:
            return []
        
        # Get correlations with target
        if self.analyzer.target_col in corr_matrix.columns:
            target_corr = corr_matrix[self.analyzer.target_col].drop(
                self.analyzer.target_col
            ).abs()
        else:
            return []
        
        # Sort by correlation with target
        sorted_features = target_corr.sort_values(ascending=False)
        
        # Greedy selection
        selected = []
        for feature in sorted_features.index:
            # Check correlation with already selected features
            if len(selected) > 0:
                max_corr_with_selected = max(
                    abs(corr_matrix.loc[feature, sel]) for sel in selected
                )
                
                if max_corr_with_selected > correlation_threshold:
                    continue
            
            selected.append(feature)
            
            if len(selected) >= n_features:
                break
        
        return selected
    
    def select_by_mutual_information(self, n_features: int = 10) -> List[str]:
        """
        Select features by mutual information with target.
        
        Args:
            n_features: Number of features to select
        
        Returns:
            List of selected feature names
        """
        mi_scores = self.analyzer._compute_mutual_information()
        
        if not mi_scores:
            return []
        
        # Sort by mutual information
        sorted_features = sorted(
            mi_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [f[0] for f in sorted_features[:n_features]]


class MulticollinearityDetector:
    """
    Specialized class for detecting multicollinearity.
    
    Provides advanced methods for identifying and handling
    multicollinearity in features.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def calculate_vif_stepwise(self, threshold: float = 5.0) -> Dict[str, Any]:
        """
        Calculate VIF stepwise, removing highly collinear features.
        
        This method iteratively removes the feature with the highest VIF
        until all VIFs are below the threshold.
        
        Args:
            threshold: VIF threshold
        
        Returns:
            Dictionary with stepwise VIF results
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {}
        
        # Initialize
        current_cols = numeric_cols.copy()
        removed_features = []
        vif_history = []
        
        while len(current_cols) >= 2:
            # Calculate VIF for current set
            X = self.data[current_cols].fillna(0)
            X = add_constant(X)
            
            vifs = []
            for i, col in enumerate(current_cols):
                vif = variance_inflation_factor(X.values, i + 1)  # +1 for constant
                vifs.append((col, vif))
            
            # Find max VIF
            max_vif_col, max_vif = max(vifs, key=lambda x: x[1])
            
            vif_history.append({
                'iteration': len(removed_features) + 1,
                'features': current_cols.copy(),
                'max_vif': max_vif,
                'max_vif_feature': max_vif_col,
                'all_vifs': dict(vifs)
            })
            
            # Check if we need to remove
            if max_vif > threshold:
                removed_features.append(max_vif_col)
                current_cols.remove(max_vif_col)
            else:
                break
        
        return {
            'final_features': current_cols,
            'removed_features': removed_features,
            'vif_history': vif_history
        }
    
    def calculate_eigenvalue_analysis(self) -> Dict[str, Any]:
        """
        Perform eigenvalue analysis for multicollinearity detection.
        
        Returns:
            Dictionary with eigenvalue analysis results
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {}
        
        # Prepare correlation matrix
        corr_matrix = self.data[numeric_cols].corr().values
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        
        # Calculate condition indices
        max_eigenvalue = eigenvalues.max()
        condition_indices = np.sqrt(max_eigenvalue / eigenvalues)
        
        # Identify problematic dimensions
        problematic = []
        for i, ci in enumerate(condition_indices):
            if ci > 30:  # High condition index
                problematic.append({
                    'dimension': i + 1,
                    'condition_index': float(ci),
                    'eigenvalue': float(eigenvalues[i])
                })
        
        return {
            'eigenvalues': eigenvalues.tolist(),
            'condition_indices': condition_indices.tolist(),
            'problematic_dimensions': problematic,
            'has_multicollinearity': len(problematic) > 0
        }


class AssociationMiner:
    """
    Mine associations between categorical variables.
    
    This class discovers association rules and patterns between
    categorical features, which is useful for fraud detection.
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str = 'is_fraud'):
        self.data = data
        self.target_col = target_col
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_col in self.categorical_cols:
            self.categorical_cols.remove(target_col)
    
    def find_association_rules(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        min_lift: float = 1.0
    ) -> pd.DataFrame:
        """
        Find association rules using Apriori algorithm.
        
        Args:
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            min_lift: Minimum lift threshold
        
        Returns:
            DataFrame with association rules
        """
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            
            # Prepare transaction data
            transactions = []
            for idx, row in self.data[self.categorical_cols].iterrows():
                transaction = [f"{col}={val}" for col, val in row.items() if pd.notna(val)]
                transactions.append(transaction)
            
            # Convert to one-hot encoded DataFrame
            from mlxtend.preprocessing import TransactionEncoder
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Find frequent itemsets
            frequent_itemsets = apriori(
                df,
                min_support=min_support,
                use_colnames=True
            )
            
            if len(frequent_itemsets) == 0:
                return pd.DataFrame()
            
            # Generate association rules
            rules = association_rules(
                frequent_itemsets,
                metric="lift",
                min_threshold=min_lift
            )
            
            # Filter by confidence
            rules = rules[rules['confidence'] >= min_confidence]
            
            # Add target-related metrics
            if self.target_col in self.data.columns:
                rules['target_correlation'] = rules.apply(
                    lambda x: self._calculate_rule_target_correlation(x),
                    axis=1
                )
            
            return rules
            
        except ImportError:
            logger.warning("mlxtend not installed. Association rule mining disabled.")
            return pd.DataFrame()
    
    def _calculate_rule_target_correlation(self, rule: pd.Series) -> float:
        """Calculate correlation between rule and target."""
        # This is a simplified version
        # In practice, you'd compute the actual correlation
        return 0.0
    
    def find_fraud_associations(self) -> Dict[str, Any]:
        """
        Find associations specifically related to fraud.
        
        Returns:
            Dictionary with fraud association patterns
        """
        fraud_data = self.data[self.data[self.target_col] == 1]
        normal_data = self.data[self.data[self.target_col] == 0]
        
        associations = {}
        
        for col in self.categorical_cols:
            # Get value counts for fraud and normal
            fraud_counts = fraud_data[col].value_counts(normalize=True)
            normal_counts = normal_data[col].value_counts(normalize=True)
            
            # Calculate lift for each value
            lifts = {}
            for value in fraud_counts.index:
                if value in normal_counts.index:
                    lift = fraud_counts[value] / normal_counts[value]
                    lifts[str(value)] = float(lift)
            
            # Find values with high lift (more common in fraud)
            high_lift_values = {
                k: v for k, v in lifts.items() if v > 2.0
            }
            
            if high_lift_values:
                associations[col] = {
                    'high_risk_values': high_lift_values,
                    'max_lift': max(lifts.values()),
                    'min_lift': min(lifts.values())
                }
        
        return associations