"""
Temporal Analysis Module for Fraud Detection
=============================================
This module provides comprehensive temporal analysis tools for understanding
time-based patterns in fraud detection data.

Key Features:
1. Time-based aggregations and statistics
2. Seasonality detection (hourly, daily, weekly, monthly)
3. Trend analysis and decomposition
4. Velocity calculations (transaction rates, spending velocity)
5. Time gap analysis between transactions
6. Anomaly detection in temporal patterns

The module helps identify:
- Peak fraud hours/days
- Seasonal fraud patterns
- Unusual transaction timing
- Rapid succession of transactions
- Time-based risk factors
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import logging
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

# Configure logging
logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Main class for temporal analysis of fraud data.
    
    This class provides comprehensive methods for analyzing
    time-based patterns in transaction data.
    
    Example:
        >>> analyzer = TemporalAnalyzer(df, time_col='Time', target_col='is_fraud')
        >>> patterns = analyzer.analyze_temporal_patterns()
        >>> seasonality = analyzer.detect_seasonality()
        >>> velocities = analyzer.calculate_velocities()
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        time_col: str = 'Time',
        target_col: str = 'is_fraud',
        customer_id_col: Optional[str] = None,
        amount_col: Optional[str] = 'Amount'
    ):
        """
        Initialize the temporal analyzer.
        
        Args:
            data: Input DataFrame
            time_col: Column containing time information
            target_col: Column containing fraud indicator
            customer_id_col: Column with customer identifiers
            amount_col: Column with transaction amounts
        """
        self.data = data.copy()
        self.time_col = time_col
        self.target_col = target_col
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        
        # Convert time to datetime if needed
        self._prepare_time_column()
        
        # Extract time components
        self._extract_time_components()
        
        # Separate fraud and normal data
        self.fraud_data = self.data[self.data[target_col] == 1]
        self.normal_data = self.data[self.data[target_col] == 0]
        
        logger.info(f"Initialized temporal analyzer with time range: "
                   f"{self.data['datetime'].min()} to {self.data['datetime'].max()}")
    
    def _prepare_time_column(self) -> None:
        """Prepare time column for analysis."""
        if self.time_col not in self.data.columns:
            raise ValueError(f"Time column '{self.time_col}' not found")
        
        # Convert to datetime
        if self.data[self.time_col].dtype in [np.int64, np.float64]:
            # Assume seconds since epoch
            self.data['datetime'] = pd.to_datetime(self.data[self.time_col], unit='s')
        else:
            self.data['datetime'] = pd.to_datetime(self.data[self.time_col])
        
        # Sort by time
        self.data = self.data.sort_values('datetime')
    
    def _extract_time_components(self) -> None:
        """Extract various time components."""
        self.data['hour'] = self.data['datetime'].dt.hour
        self.data['day'] = self.data['datetime'].dt.day
        self.data['day_of_week'] = self.data['datetime'].dt.dayofweek
        self.data['day_name'] = self.data['datetime'].dt.day_name()
        self.data['week'] = self.data['datetime'].dt.isocalendar().week
        self.data['month'] = self.data['datetime'].dt.month
        self.data['year'] = self.data['datetime'].dt.year
        self.data['quarter'] = self.data['datetime'].dt.quarter
        self.data['is_weekend'] = (self.data['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour
        self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        
        # Cyclical encoding for day of week
        self.data['dow_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['dow_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """
        Analyze temporal patterns in fraud data.
        
        Returns:
            Dictionary with temporal analysis results
        """
        results = {}
        
        # 1. Hourly patterns
        results['hourly'] = self._analyze_hourly_patterns()
        
        # 2. Daily patterns
        results['daily'] = self._analyze_daily_patterns()
        
        # 3. Weekly patterns
        results['weekly'] = self._analyze_weekly_patterns()
        
        # 4. Monthly patterns
        results['monthly'] = self._analyze_monthly_patterns()
        
        # 5. Weekend vs weekday
        results['weekend_analysis'] = self._analyze_weekend_patterns()
        
        # 6. Time gap analysis
        results['time_gaps'] = self._analyze_time_gaps()
        
        # 7. Peak detection
        results['peaks'] = self._detect_peaks()
        
        return results
    
    def _analyze_hourly_patterns(self) -> Dict[str, Any]:
        """Analyze hourly patterns in fraud."""
        # Fraud counts by hour
        fraud_by_hour = self.fraud_data.groupby('hour').size()
        total_by_hour = self.data.groupby('hour').size()
        
        # Fraud rate by hour
        fraud_rate_by_hour = (fraud_by_hour / total_by_hour * 100).fillna(0)
        
        # Find peak hours
        peak_hours = fraud_by_hour.nlargest(3).index.tolist()
        safest_hours = fraud_by_hour.nsmallest(3).index.tolist()
        
        # Statistical test: are frauds uniformly distributed across hours?
        if len(fraud_by_hour) > 0:
            observed = fraud_by_hour.values
            expected = [fraud_by_hour.sum() / 24] * len(observed)
            chi2, p_value = stats.chisquare(observed, expected)
            uniform_distribution = p_value > 0.05
        else:
            chi2, p_value, uniform_distribution = 0, 1, True
        
        return {
            'fraud_counts': fraud_by_hour.to_dict(),
            'fraud_rates': fraud_rate_by_hour.to_dict(),
            'peak_hours': peak_hours,
            'safest_hours': safest_hours,
            'max_fraud_hour': int(fraud_by_hour.idxmax()) if len(fraud_by_hour) > 0 else None,
            'min_fraud_hour': int(fraud_by_hour.idxmin()) if len(fraud_by_hour) > 0 else None,
            'chi_square_test': {
                'statistic': float(chi2),
                'p_value': float(p_value),
                'uniform_distribution': uniform_distribution
            }
        }
    
    def _analyze_daily_patterns(self) -> Dict[str, Any]:
        """Analyze daily patterns in fraud."""
        # Fraud counts by day of week
        fraud_by_day = self.fraud_data.groupby('day_of_week').size()
        total_by_day = self.data.groupby('day_of_week').size()
        
        # Fraud rate by day
        fraud_rate_by_day = (fraud_by_day / total_by_day * 100).fillna(0)
        
        # Day names mapping
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        
        # Find most fraudulent day
        most_fraudulent_day = day_names[int(fraud_by_day.idxmax())] if len(fraud_by_day) > 0 else None
        least_fraudulent_day = day_names[int(fraud_by_day.idxmin())] if len(fraud_by_day) > 0 else None
        
        return {
            'fraud_counts': {day_names[int(k)]: int(v) for k, v in fraud_by_day.items()},
            'fraud_rates': {day_names[int(k)]: float(v) for k, v in fraud_rate_by_day.items()},
            'most_fraudulent_day': most_fraudulent_day,
            'least_fraudulent_day': least_fraudulent_day,
            'weekend_risk_ratio': self._calculate_weekend_risk_ratio()
        }
    
    def _analyze_weekly_patterns(self) -> Dict[str, Any]:
        """Analyze weekly patterns in fraud."""
        # Fraud by week
        fraud_by_week = self.fraud_data.groupby('week').size()
        total_by_week = self.data.groupby('week').size()
        
        # Fraud rate by week
        fraud_rate_by_week = (fraud_by_week / total_by_week * 100).fillna(0)
        
        # Trend direction
        if len(fraud_rate_by_week) > 1:
            from scipy import stats
            x = np.arange(len(fraud_rate_by_week))
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x, fraud_rate_by_week.values
            )
            trend_direction = 'increasing' if slope > 0 else 'decreasing'
        else:
            slope, p_value, trend_direction = 0, 1, 'stable'
        
        return {
            'weekly_fraud_counts': fraud_by_week.to_dict(),
            'weekly_fraud_rates': fraud_rate_by_week.to_dict(),
            'trend': {
                'slope': float(slope),
                'p_value': float(p_value),
                'direction': trend_direction,
                'significant': p_value < 0.05
            }
        }
    
    def _analyze_monthly_patterns(self) -> Dict[str, Any]:
        """Analyze monthly patterns in fraud."""
        # Fraud by month
        fraud_by_month = self.fraud_data.groupby('month').size()
        total_by_month = self.data.groupby('month').size()
        
        # Fraud rate by month
        fraud_rate_by_month = (fraud_by_month / total_by_month * 100).fillna(0)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        return {
            'fraud_counts': {month_names[int(k)-1]: int(v) for k, v in fraud_by_month.items()},
            'fraud_rates': {month_names[int(k)-1]: float(v) for k, v in fraud_rate_by_month.items()}
        }
    
    def _analyze_weekend_patterns(self) -> Dict[str, Any]:
        """Compare fraud patterns between weekdays and weekends."""
        weekend_fraud = self.fraud_data[self.fraud_data['is_weekend'] == 1]
        weekday_fraud = self.fraud_data[self.fraud_data['is_weekend'] == 0]
        
        weekend_total = self.data[self.data['is_weekend'] == 1]
        weekday_total = self.data[self.data['is_weekend'] == 0]
        
        # Fraud rates
        weekend_rate = len(weekend_fraud) / len(weekend_total) * 100 if len(weekend_total) > 0 else 0
        weekday_rate = len(weekday_fraud) / len(weekday_total) * 100 if len(weekday_total) > 0 else 0
        
        # Risk ratio
        risk_ratio = weekend_rate / weekday_rate if weekday_rate > 0 else 1
        
        # Statistical test
        from statsmodels.stats.proportion import proportions_ztest
        count = [len(weekend_fraud), len(weekday_fraud)]
        nobs = [len(weekend_total), len(weekday_total)]
        
        if all(n > 0 for n in nobs):
            z_stat, p_value = proportions_ztest(count, nobs)
            significant = p_value < 0.05
        else:
            z_stat, p_value, significant = 0, 1, False
        
        return {
            'weekend_fraud_rate': float(weekend_rate),
            'weekday_fraud_rate': float(weekday_rate),
            'risk_ratio': float(risk_ratio),
            'interpretation': f"Weekends are {risk_ratio:.2f}x riskier than weekdays",
            'statistical_test': {
                'z_statistic': float(z_stat),
                'p_value': float(p_value),
                'significant': significant
            }
        }
    
    def _calculate_weekend_risk_ratio(self) -> float:
        """Calculate the ratio of weekend fraud rate to weekday fraud rate."""
        weekend_rate = self.fraud_data[self.fraud_data['is_weekend'] == 1].shape[0] / \
                      max(1, self.data[self.data['is_weekend'] == 1].shape[0])
        
        weekday_rate = self.fraud_data[self.fraud_data['is_weekend'] == 0].shape[0] / \
                      max(1, self.data[self.data['is_weekend'] == 0].shape[0])
        
        return weekend_rate / weekday_rate if weekday_rate > 0 else 1
    
    def _analyze_time_gaps(self) -> Dict[str, Any]:
        """Analyze time gaps between transactions."""
        if self.customer_id_col is None:
            logger.warning("Customer ID column not provided. Skipping time gap analysis.")
            return {}
        
        # Calculate time gaps for each customer
        self.data['prev_time'] = self.data.groupby(self.customer_id_col)['datetime'].shift(1)
        self.data['time_gap'] = (self.data['datetime'] - self.data['prev_time']).dt.total_seconds()
        
        # Separate fraud and normal time gaps
        fraud_gaps = self.data[self.data[self.target_col] == 1]['time_gap'].dropna()
        normal_gaps = self.data[self.data[self.target_col] == 0]['time_gap'].dropna()
        
        if len(fraud_gaps) == 0 or len(normal_gaps) == 0:
            return {}
        
        # Statistical comparison
        from scipy.stats import mannwhitneyu
        u_stat, p_value = mannwhitneyu(fraud_gaps, normal_gaps, alternative='two-sided')
        
        # Calculate percentiles
        fraud_percentiles = {
            'p25': float(fraud_gaps.quantile(0.25)),
            'p50': float(fraud_gaps.quantile(0.5)),
            'p75': float(fraud_gaps.quantile(0.75)),
            'p90': float(fraud_gaps.quantile(0.9)),
            'p95': float(fraud_gaps.quantile(0.95))
        }
        
        normal_percentiles = {
            'p25': float(normal_gaps.quantile(0.25)),
            'p50': float(normal_gaps.quantile(0.5)),
            'p75': float(normal_gaps.quantile(0.75)),
            'p90': float(normal_gaps.quantile(0.9)),
            'p95': float(normal_gaps.quantile(0.95))
        }
        
        return {
            'fraud_time_gaps': fraud_percentiles,
            'normal_time_gaps': normal_percentiles,
            'comparison': {
                'mann_whitney_u': float(u_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': 'Fraud has significantly different time gaps' if p_value < 0.05 else 'No significant difference'
            },
            'rapid_succession_risk': {
                'fraud_rapid_rate': (fraud_gaps < 60).mean(),  # % of fraud within 1 minute
                'normal_rapid_rate': (normal_gaps < 60).mean()
            }
        }
    
    def _detect_peaks(self) -> Dict[str, Any]:
        """Detect peaks in fraud time series."""
        # Resample to hourly
        hourly_fraud = self.data.set_index('datetime').resample('H')[self.target_col].sum()
        
        # Detect peaks
        peaks, properties = find_peaks(
            hourly_fraud.values,
            height=hourly_fraud.quantile(0.9),  # Only peaks above 90th percentile
            distance=24  # At least 24 hours between peaks
        )
        
        peak_times = hourly_fraud.index[peaks]
        peak_values = hourly_fraud.values[peaks]
        
        return {
            'n_peaks': len(peaks),
            'peak_times': [str(t) for t in peak_times],
            'peak_values': peak_values.tolist(),
            'average_peak_height': float(peak_values.mean()) if len(peak_values) > 0 else 0,
            'peak_density': len(peaks) / len(hourly_fraud) * 24  # Peaks per day
        }
    
    def detect_seasonality(self) -> Dict[str, Any]:
        """
        Detect seasonality patterns in fraud data.
        
        Returns:
            Dictionary with seasonality detection results
        """
        results = {}
        
        # Resample to daily
        daily_fraud = self.data.set_index('datetime').resample('D')[self.target_col].sum()
        daily_fraud = daily_fraud.fillna(0)
        
        if len(daily_fraud) < 14:  # Need at least 2 weeks
            logger.warning("Insufficient data for seasonality detection")
            return results
        
        # 1. Autocorrelation analysis
        results['autocorrelation'] = self._analyze_autocorrelation(daily_fraud)
        
        # 2. Seasonal decomposition
        results['decomposition'] = self._seasonal_decomposition(daily_fraud)
        
        # 3. Stationarity tests
        results['stationarity'] = self._test_stationarity(daily_fraud)
        
        return results
    
    def _analyze_autocorrelation(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze autocorrelation to detect seasonality."""
        from statsmodels.graphics.tsaplots import plot_acf
        from statsmodels.tsa.stattools import acf
        
        # Calculate autocorrelation
        lags = min(30, len(series) // 2)
        autocorr = acf(series, nlags=lags, fft=True)
        
        # Find significant lags
        confidence_interval = 1.96 / np.sqrt(len(series))
        significant_lags = []
        for i, corr in enumerate(autocorr[1:], 1):  # Skip lag 0
            if abs(corr) > confidence_interval:
                significant_lags.append(i)
        
        # Detect seasonality periods
        periods = []
        for lag in significant_lags:
            if 6 < lag < 8:  # Weekly pattern (7 days)
                periods.append('weekly')
            elif 27 < lag < 31:  # Monthly pattern
                periods.append('monthly')
        
        return {
            'autocorrelations': autocorr.tolist(),
            'significant_lags': significant_lags,
            'detected_periods': list(set(periods)),
            'has_seasonality': len(periods) > 0
        }
    
    def _seasonal_decomposition(self, series: pd.Series) -> Dict[str, Any]:
        """Perform seasonal decomposition."""
        try:
            # Determine period
            if len(series) >= 30:
                period = 7  # Weekly seasonality
            else:
                period = 1
            
            decomposition = seasonal_decompose(
                series,
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )
            
            # Calculate strength of seasonality
            seasonal_strength = 1 - (decomposition.resid.var() / 
                                     (decomposition.seasonal + decomposition.resid).var())
            
            # Calculate strength of trend
            trend_strength = 1 - (decomposition.resid.var() / 
                                  (decomposition.trend + decomposition.resid).var())
            
            return {
                'has_seasonality': seasonal_strength > 0.3,
                'seasonal_strength': float(seasonal_strength),
                'trend_strength': float(trend_strength),
                'seasonal_component': decomposition.seasonal.tolist(),
                'trend_component': decomposition.trend.tolist(),
                'residual_component': decomposition.resid.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Seasonal decomposition failed: {str(e)}")
            return {}
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Test for stationarity in time series."""
        results = {}
        
        # Augmented Dickey-Fuller test
        try:
            adf_stat, adf_p, _, _, critical_values, _ = adfuller(series.dropna())
            results['adf_test'] = {
                'statistic': float(adf_stat),
                'p_value': float(adf_p),
                'is_stationary': adf_p < 0.05,
                'critical_values': critical_values
            }
        except Exception as e:
            logger.warning(f"ADF test failed: {str(e)}")
        
        # KPSS test
        try:
            kpss_stat, kpss_p, _, critical_values = kpss(series.dropna(), regression='c')
            results['kpss_test'] = {
                'statistic': float(kpss_stat),
                'p_value': float(kpss_p),
                'is_stationary': kpss_p > 0.05,
                'critical_values': critical_values
            }
        except Exception as e:
            logger.warning(f"KPSS test failed: {str(e)}")
        
        return results


class SeasonalityDetector:
    """
    Specialized class for detecting seasonality patterns.
    
    Provides advanced methods for identifying and quantifying
    seasonal patterns in fraud data.
    """
    
    def __init__(self, series: pd.Series):
        """
        Initialize with time series data.
        
        Args:
            series: Time series with datetime index
        """
        self.series = series
    
    def detect_multiple_seasonalities(self) -> Dict[str, Any]:
        """
        Detect multiple seasonality periods.
        
        Returns:
            Dictionary with multiple seasonality detection results
        """
        from statsmodels.tsa.stattools import acf
        
        results = {}
        
        # Calculate autocorrelation for different periods
        for period_name, period_length in [('hourly', 24), ('daily', 24*7), ('weekly', 24*7*4)]:
            if len(self.series) >= period_length * 2:
                # Resample if needed
                if period_name == 'hourly' and self.series.index.freq != 'H':
                    resampled = self.series.resample('H').sum()
                elif period_name == 'daily' and self.series.index.freq != 'D':
                    resampled = self.series.resample('D').sum()
                elif period_name == 'weekly' and self.series.index.freq != 'W':
                    resampled = self.series.resample('W').sum()
                else:
                    resampled = self.series
                
                # Calculate autocorrelation at seasonal lags
                autocorr = acf(resampled, nlags=period_length, fft=True)
                
                # Check if seasonal lag is significant
                seasonal_lag = period_length
                if seasonal_lag < len(autocorr):
                    seasonal_corr = autocorr[seasonal_lag]
                    confidence = 1.96 / np.sqrt(len(resampled))
                    
                    results[period_name] = {
                        'detected': abs(seasonal_corr) > confidence,
                        'correlation': float(seasonal_corr),
                        'strength': 'strong' if abs(seasonal_corr) > 0.5 else 'moderate' if abs(seasonal_corr) > 0.3 else 'weak'
                    }
        
        return results


class TrendAnalyzer:
    """
    Specialized class for trend analysis.
    
    Provides methods for detecting and quantifying trends
    in fraud time series.
    """
    
    def __init__(self, series: pd.Series):
        self.series = series
    
    def analyze_trend(self) -> Dict[str, Any]:
        """
        Analyze trend in time series.
        
        Returns:
            Dictionary with trend analysis results
        """
        # Create time index
        x = np.arange(len(self.series))
        y = self.series.values
        
        # Linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Exponential trend
        try:
            log_y = np.log(y[y > 0])
            log_x = x[y > 0]
            exp_slope, exp_intercept, exp_r, exp_p, exp_err = stats.linregress(log_x, log_y)
            exp_growth_rate = np.exp(exp_slope) - 1
        except:
            exp_growth_rate = 0
        
        # Moving average
        ma_7 = pd.Series(y).rolling(window=7, min_periods=1).mean()
        ma_30 = pd.Series(y).rolling(window=30, min_periods=1).mean()
        
        # Detect change points
        from ruptures import Pelt, Binseg
        try:
            # PELT algorithm for change point detection
            model = Pelt(model="rbf").fit(y.reshape(-1, 1))
            change_points = model.predict(pen=10)
        except:
            change_points = []
        
        return {
            'linear_trend': {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'direction': 'increasing' if slope > 0 else 'decreasing'
            },
            'exponential_growth_rate': float(exp_growth_rate),
            'moving_averages': {
                '7_day': ma_7.tolist(),
                '30_day': ma_30.tolist()
            },
            'change_points': [int(cp) for cp in change_points if cp < len(y)]
        }


class VelocityCalculator:
    """
    Calculate various velocity metrics for fraud detection.
    
    Velocity metrics measure the rate of change in transaction
    behavior, which is often indicative of fraud.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        customer_id_col: str,
        time_col: str = 'datetime',
        amount_col: Optional[str] = None
    ):
        self.data = data
        self.customer_id_col = customer_id_col
        self.time_col = time_col
        self.amount_col = amount_col
    
    def calculate_all_velocities(self) -> pd.DataFrame:
        """
        Calculate all velocity metrics.
        
        Returns:
            DataFrame with velocity features added
        """
        df = self.data.copy()
        
        # Transaction velocity
        df = self._calculate_transaction_velocity(df)
        
        # Amount velocity
        if self.amount_col:
            df = self._calculate_amount_velocity(df)
        
        # Location velocity (if available)
        if 'location' in df.columns:
            df = self._calculate_location_velocity(df)
        
        return df
    
    def _calculate_transaction_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate transaction velocity metrics."""
        # Sort by customer and time
        df = df.sort_values([self.customer_id_col, self.time_col])
        
        # Rolling counts for different windows
        for window in ['1H', '24H', '7D']:
            col_name = f'tx_velocity_{window}'
            df[col_name] = df.groupby(self.customer_id_col)[self.time_col].transform(
                lambda x: x.rolling(window, min_periods=1).count()
            )
        
        # Calculate acceleration (change in velocity)
        df['tx_acceleration'] = df.groupby(self.customer_id_col)['tx_velocity_1H'].diff()
        
        return df
    
    def _calculate_amount_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate amount velocity metrics."""
        # Rolling sums for different windows
        for window in ['1H', '24H', '7D']:
            col_name = f'amount_velocity_{window}'
            df[col_name] = df.groupby(self.customer_id_col)[self.amount_col].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )
        
        # Average transaction amount
        df['avg_amount_24h'] = df.groupby(self.customer_id_col)['amount_velocity_24H'].transform(
            lambda x: x / x.rolling('24H', min_periods=1).count()
        )
        
        return df
    
    def _calculate_location_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate location change velocity."""
        # Count unique locations in time windows
        def unique_locations_count(x):
            return x.nunique()
        
        df['location_changes_24h'] = df.groupby(self.customer_id_col)['location'].transform(
            lambda x: x.rolling('24H', min_periods=1).apply(unique_locations_count)
        )
        
        return df
    
    def detect_velocity_anomalies(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies in velocity metrics.
        
        Args:
            threshold: Z-score threshold for anomaly detection
        
        Returns:
            DataFrame with velocity anomaly flags
        """
        df = self.calculate_all_velocities()
        
        velocity_cols = [col for col in df.columns if 'velocity' in col]
        
        for col in velocity_cols:
            # Calculate z-scores
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:
                z_scores = (df[col] - mean) / std
                df[f'{col}_anomaly'] = (abs(z_scores) > threshold).astype(int)
        
        return df


class TimeGapAnalyzer:
    """
    Analyze time gaps between transactions.
    
    Time gaps are crucial for detecting:
    - Rapid succession fraud
    - Unusual waiting periods
    - Bot-like behavior
    """
    
    def __init__(self, data: pd.DataFrame, customer_id_col: str, time_col: str = 'datetime'):
        self.data = data
        self.customer_id_col = customer_id_col
        self.time_col = time_col
    
    def calculate_time_gaps(self) -> pd.DataFrame:
        """
        Calculate time gaps between consecutive transactions.
        
        Returns:
            DataFrame with time gap features
        """
        df = self.data.copy()
        
        # Sort by customer and time
        df = df.sort_values([self.customer_id_col, self.time_col])
        
        # Calculate time since last transaction
        df['time_since_last'] = df.groupby(self.customer_id_col)[self.time_col].diff()
        
        # Convert to seconds
        df['time_gap_seconds'] = df['time_since_last'].dt.total_seconds()
        
        # Time to next transaction
        df['time_to_next'] = df.groupby(self.customer_id_col)[self.time_col].diff(-1).dt.total_seconds().abs()
        
        # Time-based features
        df['is_rapid'] = (df['time_gap_seconds'] < 60).astype(int)  # Less than 1 minute
        df['is_very_rapid'] = (df['time_gap_seconds'] < 10).astype(int)  # Less than 10 seconds
        
        # Rolling statistics of time gaps
        for window in [3, 5, 10]:
            df[f'avg_time_gap_last_{window}'] = df.groupby(self.customer_id_col)['time_gap_seconds'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'std_time_gap_last_{window}'] = df.groupby(self.customer_id_col)['time_gap_seconds'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        return df
    
    def detect_burst_patterns(self) -> Dict[str, Any]:
        """
        Detect burst patterns (rapid succession of transactions).
        
        Returns:
            Dictionary with burst pattern analysis
        """
        df = self.calculate_time_gaps()
        
        # Define burst: 3+ transactions within 5 minutes
        burst_threshold = 300  # 5 minutes in seconds
        
        bursts = []
        for customer in df[self.customer_id_col].unique():
            customer_data = df[df[self.customer_id_col] == customer]
            
            # Find sequences of rapid transactions
            rapid_mask = customer_data['time_gap_seconds'] < burst_threshold
            
            # Identify burst starts
            burst_starts = rapid_mask & ~rapid_mask.shift(1).fillna(False)
            
            for idx in customer_data[burst_starts].index:
                burst = [idx]
                current_idx = idx
                
                # Add subsequent rapid transactions
                while current_idx in customer_data.index and \
                      customer_data.loc[current_idx, 'time_gap_seconds'] < burst_threshold:
                    burst.append(current_idx)
                    next_idx = customer_data.index.get_loc(current_idx) + 1
                    if next_idx < len(customer_data):
                        current_idx = customer_data.index[next_idx]
                    else:
                        break
                
                if len(burst) >= 3:
                    bursts.append({
                        'customer': customer,
                        'start_time': customer_data.loc[burst[0], self.time_col],
                        'end_time': customer_data.loc[burst[-1], self.time_col],
                        'transaction_count': len(burst),
                        'total_amount': customer_data.loc[burst, 'Amount'].sum() if 'Amount' in customer_data.columns else 0,
                        'fraud_count': customer_data.loc[burst, 'is_fraud'].sum() if 'is_fraud' in customer_data.columns else 0
                    })
        
        # Analyze burst characteristics
        if bursts:
            burst_df = pd.DataFrame(bursts)
            analysis = {
                'total_bursts': len(bursts),
                'avg_burst_size': float(burst_df['transaction_count'].mean()),
                'max_burst_size': int(burst_df['transaction_count'].max()),
                'bursts_with_fraud': int((burst_df['fraud_count'] > 0).sum()),
                'fraud_burst_rate': float((burst_df['fraud_count'] > 0).mean()),
                'avg_burst_amount': float(burst_df['total_amount'].mean()) if 'total_amount' in burst_df.columns else 0
            }
        else:
            analysis = {
                'total_bursts': 0,
                'avg_burst_size': 0,
                'max_burst_size': 0,
                'bursts_with_fraud': 0,
                'fraud_burst_rate': 0,
                'avg_burst_amount': 0
            }
        
        return analysis