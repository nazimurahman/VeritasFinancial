"""
Statistical Analysis Module for Fraud Detection
================================================
This module provides comprehensive statistical analysis tools specifically
designed for banking fraud detection. It includes:

1. Descriptive Statistics: Mean, median, mode, variance, etc.
2. Hypothesis Testing: T-tests, chi-square, ANOVA for fraud patterns
3. Distribution Analysis: Normal distribution tests, probability fitting
4. Quality Assessment: Missing data, outliers, data integrity checks

The module handles the unique challenges of fraud data:
- Severe class imbalance
- Skewed distributions
- Time-dependent patterns
- Complex relationships between features
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    norm, chi2, ttest_ind, mannwhitneyu, 
    shapiro, anderson, ks_2samp, f_oneway
)
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class StatisticalResults:
    """
    Data class to store statistical analysis results.
    
    This class provides a structured way to store and access
    results from various statistical analyses.
    
    Attributes:
        descriptive_stats: Basic statistical measures
        hypothesis_tests: Results of statistical tests
        distributions: Fitted distribution parameters
        quality_metrics: Data quality indicators
        insights: Key findings and interpretations
    """
    descriptive_stats: Dict[str, Any] = field(default_factory=dict)
    hypothesis_tests: Dict[str, Any] = field(default_factory=dict)
    distributions: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)


class FraudStatisticalAnalysis:
    """
    Main class for comprehensive statistical analysis of fraud data.
    
    This class orchestrates all statistical analyses, providing a unified
    interface for exploring fraud patterns in banking transactions.
    
    Example:
        >>> analyzer = FraudStatisticalAnalysis(df, target_col='is_fraud')
        >>> results = analyzer.run_complete_analysis()
        >>> print(results.insights)
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        target_col: str = 'is_fraud',
        config: Optional[Dict] = None
    ):
        """
        Initialize the statistical analyzer.
        
        Args:
            data: Input DataFrame containing transaction data
            target_col: Name of the target column (fraud indicator)
            config: Optional configuration dictionary for customizing analysis
        
        Raises:
            ValueError: If target column not found in data
        """
        # Validate input data
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        self.data = data.copy()
        self.target_col = target_col
        self.config = config or {}
        
        # Split data by class for comparative analysis
        self.fraud_data = data[data[target_col] == 1]
        self.normal_data = data[data[target_col] == 0]
        
        # Separate numeric and categorical columns
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from feature lists
        if target_col in self.numeric_cols:
            self.numeric_cols.remove(target_col)
        
        logger.info(f"Initialized analyzer with {len(data)} rows, {len(self.numeric_cols)} numeric features")
    
    def run_complete_analysis(self) -> StatisticalResults:
        """
        Execute comprehensive statistical analysis pipeline.
        
        This method runs all analyses in sequence and compiles results
        into a single StatisticalResults object.
        
        Returns:
            StatisticalResults object containing all analysis results
        
        Example:
            >>> results = analyzer.run_complete_analysis()
            >>> print(f"Found {results.quality_metrics['missing_percentage']}% missing data")
        """
        logger.info("Starting comprehensive statistical analysis")
        
        results = StatisticalResults()
        
        # 1. Data Quality Assessment
        logger.info("Phase 1: Assessing data quality")
        results.quality_metrics = self._assess_data_quality()
        
        # 2. Descriptive Statistics
        logger.info("Phase 2: Computing descriptive statistics")
        results.descriptive_stats = self._compute_descriptive_stats()
        
        # 3. Distribution Analysis
        logger.info("Phase 3: Analyzing distributions")
        results.distributions = self._analyze_distributions()
        
        # 4. Hypothesis Testing
        logger.info("Phase 4: Performing hypothesis tests")
        results.hypothesis_tests = self._perform_hypothesis_tests()
        
        # 5. Generate Insights
        logger.info("Phase 5: Generating insights")
        results.insights = self._generate_insights(results)
        
        logger.info("Statistical analysis completed successfully")
        return results
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """
        Assess the quality of input data.
        
        This method checks for:
        - Missing values and their patterns
        - Data types consistency
        - Zero values (potential data entry issues)
        - Duplicate records
        - Memory usage
        
        Returns:
            Dictionary containing quality metrics
        """
        quality_metrics = {}
        
        # 1. Missing Value Analysis
        missing_count = self.data.isnull().sum()
        missing_percentage = (missing_count / len(self.data)) * 100
        
        quality_metrics['missing_count'] = missing_count.to_dict()
        quality_metrics['missing_percentage'] = missing_percentage.to_dict()
        quality_metrics['total_missing'] = self.data.isnull().sum().sum()
        quality_metrics['rows_with_missing'] = self.data.isnull().any(axis=1).sum()
        
        # 2. Missing Pattern Analysis
        # Check if missing values occur in specific patterns
        missing_correlation = self.data.isnull().corr()
        quality_metrics['missing_correlation'] = missing_correlation.to_dict() if not missing_correlation.empty else {}
        
        # 3. Zero Value Analysis (for numeric columns)
        zero_counts = {}
        for col in self.numeric_cols:
            zero_count = (self.data[col] == 0).sum()
            if zero_count > 0:
                zero_counts[col] = {
                    'count': int(zero_count),
                    'percentage': float(zero_count / len(self.data) * 100)
                }
        quality_metrics['zero_values'] = zero_counts
        
        # 4. Data Type Consistency
        dtype_counts = self.data.dtypes.value_counts().to_dict()
        quality_metrics['data_types'] = {str(k): int(v) for k, v in dtype_counts.items()}
        
        # 5. Duplicate Detection
        duplicate_rows = self.data.duplicated().sum()
        quality_metrics['duplicate_rows'] = int(duplicate_rows)
        quality_metrics['duplicate_percentage'] = float(duplicate_rows / len(self.data) * 100)
        
        # 6. Memory Usage
        memory_usage = self.data.memory_usage(deep=True).sum()
        quality_metrics['memory_usage_bytes'] = int(memory_usage)
        quality_metrics['memory_usage_mb'] = float(memory_usage / (1024 * 1024))
        
        return quality_metrics
    
    def _compute_descriptive_stats(self) -> Dict[str, Any]:
        """
        Compute comprehensive descriptive statistics.
        
        Calculates:
        - Basic statistics (mean, median, std, etc.)
        - Percentiles for understanding distributions
        - Skewness and kurtosis
        - Statistics split by fraud/normal classes
        - Range and IQR for outlier detection
        
        Returns:
            Dictionary with descriptive statistics
        """
        descriptive = {}
        
        # 1. Overall Statistics
        overall_stats = self.data[self.numeric_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
        descriptive['overall'] = overall_stats.to_dict()
        
        # 2. Statistics by Class (Fraud vs Normal)
        class_stats = {}
        for class_label, class_data in [('fraud', self.fraud_data), ('normal', self.normal_data)]:
            if len(class_data) > 0:
                stats_df = class_data[self.numeric_cols].describe(percentiles=[.25, .5, .75])
                class_stats[class_label] = stats_df.to_dict()
        descriptive['by_class'] = class_stats
        
        # 3. Skewness and Kurtosis
        skewness = {}
        kurtosis = {}
        for col in self.numeric_cols:
            # Calculate skewness (measure of asymmetry)
            skewness[col] = float(self.data[col].skew())
            # Calculate kurtosis (measure of tailedness)
            kurtosis[col] = float(self.data[col].kurtosis())
        
        descriptive['skewness'] = skewness
        descriptive['kurtosis'] = kurtosis
        
        # 4. Range and IQR Analysis
        range_stats = {}
        for col in self.numeric_cols:
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            
            range_stats[col] = {
                'min': float(self.data[col].min()),
                'max': float(self.data[col].max()),
                'range': float(self.data[col].max() - self.data[col].min()),
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr),
                'lower_bound': float(q1 - 1.5 * iqr),  # Outlier lower bound
                'upper_bound': float(q3 + 1.5 * iqr)   # Outlier upper bound
            }
        descriptive['range_analysis'] = range_stats
        
        # 5. Coefficient of Variation (relative variability)
        cv_stats = {}
        for col in self.numeric_cols:
            mean = self.data[col].mean()
            std = self.data[col].std()
            if mean != 0:
                cv = std / mean
                cv_stats[col] = float(cv)
        descriptive['coefficient_variation'] = cv_stats
        
        return descriptive
    
    def _analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze probability distributions of features.
        
        For each numeric feature, this method:
        1. Tests for normality using multiple tests
        2. Fits common distributions (normal, exponential, etc.)
        3. Compares fraud vs normal distributions
        4. Identifies multimodal distributions
        
        Returns:
            Dictionary with distribution analysis results
        """
        distributions = {}
        
        for col in self.numeric_cols:
            col_data = self.data[col].dropna()
            
            if len(col_data) < 3:  # Skip columns with insufficient data
                continue
            
            col_analysis = {}
            
            # 1. Normality Tests
            # Shapiro-Wilk test (best for small samples)
            if len(col_data) <= 5000:  # Shapiro is limited to 5000 samples
                shapiro_stat, shapiro_p = shapiro(col_data)
                col_analysis['shapiro_test'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
            
            # Anderson-Darling test (more powerful for large samples)
            anderson_result = anderson(col_data, dist='norm')
            col_analysis['anderson_test'] = {
                'statistic': float(anderson_result.statistic),
                'critical_values': anderson_result.critical_values.tolist(),
                'significance_level': anderson_result.significance_level.tolist(),
                'is_normal': anderson_result.statistic < anderson_result.critical_values[2]  # 5% level
            }
            
            # 2. Distribution Fitting
            # Try fitting different distributions
            distributions_to_try = ['norm', 'expon', 'gamma', 'lognorm']
            fitted_dists = {}
            
            for dist_name in distributions_to_try:
                try:
                    # Get distribution object
                    dist = getattr(stats, dist_name)
                    
                    # Fit distribution parameters
                    params = dist.fit(col_data)
                    
                    # Calculate Kolmogorov-Smirnov test
                    ks_stat, ks_p = ks_2samp(col_data, dist.rvs(*params, size=len(col_data)))
                    
                    fitted_dists[dist_name] = {
                        'parameters': [float(p) for p in params],
                        'ks_statistic': float(ks_stat),
                        'ks_p_value': float(ks_p),
                        'log_likelihood': float(np.sum(dist.logpdf(col_data, *params)))
                    }
                except Exception as e:
                    logger.debug(f"Could not fit {dist_name} to {col}: {str(e)}")
                    continue
            
            # Select best fitting distribution (lowest KS statistic)
            if fitted_dists:
                best_dist = min(fitted_dists.items(), key=lambda x: x[1]['ks_statistic'])
                col_analysis['best_fit'] = {
                    'distribution': best_dist[0],
                    **best_dist[1]
                }
            
            # 3. Multimodality Detection
            # Use Gaussian Mixture Model or dip test
            from scipy.stats import gaussian_kde
            
            # Kernel density estimation
            kde = gaussian_kde(col_data)
            x_grid = np.linspace(col_data.min(), col_data.max(), 100)
            density = kde(x_grid)
            
            # Find peaks in density (potential modes)
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(density, height=0.01 * density.max())
            
            col_analysis['modes'] = {
                'n_peaks': len(peaks),
                'peak_locations': x_grid[peaks].tolist(),
                'is_multimodal': len(peaks) > 1
            }
            
            # 4. Compare Fraud vs Normal Distributions
            fraud_col = self.fraud_data[col].dropna()
            normal_col = self.normal_data[col].dropna()
            
            if len(fraud_col) > 0 and len(normal_col) > 0:
                # Kolmogorov-Smirnov test to see if distributions differ
                ks_stat, ks_p = ks_2samp(fraud_col, normal_col)
                col_analysis['class_distribution_comparison'] = {
                    'ks_statistic': float(ks_stat),
                    'ks_p_value': float(ks_p),
                    'distributions_differ': ks_p < 0.05
                }
            
            distributions[col] = col_analysis
        
        return distributions
    
    def _perform_hypothesis_tests(self) -> Dict[str, Any]:
        """
        Perform statistical hypothesis tests for fraud patterns.
        
        Tests include:
        1. T-tests for comparing means between fraud and normal
        2. Mann-Whitney U for non-parametric comparison
        3. Chi-square tests for categorical variables
        4. ANOVA for multi-group comparisons
        5. Correlation tests between features
        
        Returns:
            Dictionary with hypothesis test results
        """
        hypothesis_results = {}
        
        # 1. Compare Fraud vs Normal for each numeric feature
        numeric_comparisons = {}
        for col in self.numeric_cols:
            fraud_values = self.fraud_data[col].dropna()
            normal_values = self.normal_data[col].dropna()
            
            if len(fraud_values) < 2 or len(normal_values) < 2:
                continue
            
            comparison = {}
            
            # Independent t-test (parametric)
            t_stat, t_p = ttest_ind(fraud_values, normal_values, equal_var=False)
            comparison['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'significant': t_p < 0.05,
                'interpretation': f"Fraud mean is {'higher' if t_stat > 0 else 'lower'} than normal" if t_p < 0.05 else "No significant difference"
            }
            
            # Mann-Whitney U test (non-parametric, robust to outliers)
            u_stat, u_p = mannwhitneyu(fraud_values, normal_values, alternative='two-sided')
            comparison['mann_whitney'] = {
                'statistic': float(u_stat),
                'p_value': float(u_p),
                'significant': u_p < 0.05,
                'interpretation': f"Fraud distribution is {'higher' if u_stat > len(fraud_values)*len(normal_values)/2 else 'lower'} than normal" if u_p < 0.05 else "No distribution difference"
            }
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((fraud_values.std()**2 + normal_values.std()**2) / 2)
            if pooled_std > 0:
                cohens_d = abs(fraud_values.mean() - normal_values.mean()) / pooled_std
                comparison['effect_size'] = {
                    'cohens_d': float(cohens_d),
                    'magnitude': 'small' if cohens_d < 0.2 else 'medium' if cohens_d < 0.5 else 'large' if cohens_d < 0.8 else 'very large'
                }
            
            numeric_comparisons[col] = comparison
        
        hypothesis_results['numeric_comparisons'] = numeric_comparisons
        
        # 2. Categorical Variable Analysis
        categorical_analysis = {}
        for col in self.categorical_cols:
            if col == self.target_col:
                continue
            
            # Create contingency table
            contingency = pd.crosstab(self.data[col], self.data[self.target_col])
            
            # Chi-square test of independence
            chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency)
            
            # Cramér's V (measure of association)
            n = len(self.data)
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))
            
            categorical_analysis[col] = {
                'chi_square': {
                    'statistic': float(chi2_stat),
                    'p_value': float(chi2_p),
                    'degrees_freedom': int(dof),
                    'significant': chi2_p < 0.05
                },
                'cramers_v': float(cramers_v),
                'association_strength': 'weak' if cramers_v < 0.1 else 'moderate' if cramers_v < 0.3 else 'strong',
                'expected_frequencies': expected.tolist()
            }
        
        hypothesis_results['categorical_analysis'] = categorical_analysis
        
        # 3. Correlation Tests
        correlation_tests = {}
        for i, col1 in enumerate(self.numeric_cols):
            for col2 in self.numeric_cols[i+1:]:
                # Pearson correlation
                pearson_corr, pearson_p = stats.pearsonr(
                    self.data[col1].fillna(self.data[col1].mean()),
                    self.data[col2].fillna(self.data[col2].mean())
                )
                
                # Spearman rank correlation (non-parametric)
                spearman_corr, spearman_p = stats.spearmanr(
                    self.data[col1].fillna(self.data[col1].median()),
                    self.data[col2].fillna(self.data[col2].median())
                )
                
                if abs(pearson_corr) > 0.5 or abs(spearman_corr) > 0.5:
                    correlation_tests[f"{col1}_vs_{col2}"] = {
                        'pearson': {
                            'correlation': float(pearson_corr),
                            'p_value': float(pearson_p),
                            'significant': pearson_p < 0.05
                        },
                        'spearman': {
                            'correlation': float(spearman_corr),
                            'p_value': float(spearman_p),
                            'significant': spearman_p < 0.05
                        }
                    }
        
        hypothesis_results['correlation_tests'] = correlation_tests
        
        return hypothesis_results
    
    def _generate_insights(self, results: StatisticalResults) -> List[str]:
        """
        Generate natural language insights from statistical analysis.
        
        This method interprets the statistical results and produces
        actionable insights for fraud analysts.
        
        Args:
            results: StatisticalResults object from previous analyses
        
        Returns:
            List of insight strings
        """
        insights = []
        
        # 1. Data Quality Insights
        quality = results.quality_metrics
        if quality['total_missing'] > 0:
            insights.append(f"⚠️ Data contains {quality['total_missing']} missing values across {len(quality['missing_count'])} columns")
        
        if quality.get('duplicate_rows', 0) > 0:
            insights.append(f"⚠️ Found {quality['duplicate_rows']} duplicate transactions ({quality['duplicate_percentage']:.2f}%)")
        
        # 2. Class Imbalance Insights
        fraud_ratio = len(self.fraud_data) / len(self.data) * 100
        insights.append(f"📊 Class imbalance: {fraud_ratio:.4f}% fraud transactions (ratio: {len(self.normal_data)/len(self.fraud_data):.1f}:1)")
        
        # 3. Distribution Insights
        for col, dist in results.distributions.items():
            if dist.get('modes', {}).get('is_multimodal', False):
                insights.append(f"📈 {col} shows multimodal distribution - suggests multiple transaction types")
            
            if dist.get('class_distribution_comparison', {}).get('distributions_differ', False):
                insights.append(f"🔍 {col} distribution differs significantly between fraud and normal transactions")
        
        # 4. Hypothesis Test Insights
        for col, comp in results.hypothesis_tests.get('numeric_comparisons', {}).items():
            if comp.get('t_test', {}).get('significant', False):
                insight = comp['t_test']['interpretation']
                insights.append(f"📉 {col}: {insight}")
        
        for col, cat in results.hypothesis_tests.get('categorical_analysis', {}).items():
            if cat.get('chi_square', {}).get('significant', False):
                insights.append(f"🏷️ {col} is associated with fraud (Cramér's V = {cat['cramers_v']:.3f} - {cat['association_strength']})")
        
        return insights


class DescriptiveStatistics:
    """
    Specialized class for computing descriptive statistics.
    
    This class focuses on detailed descriptive statistics with
    additional metrics relevant for fraud detection.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def compute_robust_statistics(self, column: str) -> Dict[str, float]:
        """
        Compute robust statistics resistant to outliers.
        
        Args:
            column: Column name to analyze
        
        Returns:
            Dictionary with robust statistics
        """
        values = self.data[column].dropna()
        
        stats_dict = {
            'median': float(values.median()),
            'mad': float(np.median(np.abs(values - values.median()))),  # Median Absolute Deviation
            'trimmed_mean': float(stats.trim_mean(values, 0.1)),  # 10% trimmed mean
            'winsorized_mean': float(self._winsorized_mean(values, 0.05)),
            'q1': float(values.quantile(0.25)),
            'q3': float(values.quantile(0.75))
        }
        
        return stats_dict
    
    def _winsorized_mean(self, values: pd.Series, limits: float) -> float:
        """Calculate winsorized mean (replace extremes with percentiles)."""
        lower = values.quantile(limits)
        upper = values.quantile(1 - limits)
        winsorized = values.clip(lower, upper)
        return winsorized.mean()


class HypothesisTester:
    """
    Specialized class for statistical hypothesis testing in fraud context.
    
    Provides methods for:
    - A/B testing of fraud prevention strategies
    - Before/after intervention analysis
    - Segment comparisons
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def test_intervention_effect(
        self, 
        before_data: pd.Series, 
        after_data: pd.Series,
        test_type: str = 'proportion'
    ) -> Dict[str, Any]:
        """
        Test the effect of a fraud prevention intervention.
        
        Args:
            before_data: Fraud rates before intervention
            after_data: Fraud rates after intervention
            test_type: Type of test ('proportion', 'mean', 'rate')
        
        Returns:
            Dictionary with test results
        """
        results = {}
        
        if test_type == 'proportion':
            # Test for change in fraud proportion
            n_before = len(before_data)
            n_after = len(after_data)
            
            fraud_before = before_data.sum()
            fraud_after = after_data.sum()
            
            # Two-proportion z-test
            p1 = fraud_before / n_before
            p2 = fraud_after / n_after
            p_pooled = (fraud_before + fraud_after) / (n_before + n_after)
            
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_before + 1/n_after))
            z_score = (p1 - p2) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            results = {
                'before_rate': float(p1),
                'after_rate': float(p2),
                'absolute_change': float(p2 - p1),
                'relative_change': float((p2 - p1) / p1 * 100) if p1 > 0 else None,
                'z_score': float(z_score),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': f"Fraud rate {'decreased' if p2 < p1 else 'increased'} by {abs((p2-p1)/p1*100):.2f}%"
            }
        
        return results