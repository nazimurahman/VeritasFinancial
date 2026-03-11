"""
DateTime Processor Module

This module handles transformation of datetime features for fraud detection.
Temporal features are critical for identifying patterns in transaction timing.

Features extracted:
1. Time components - hour, day, month, year, etc.
2. Cyclical encoding - sin/cos transformations for periodic features
3. Time differences - gaps between transactions
4. Business hours - flags for working hours
5. Holiday detection - special day flags
6. Seasonal features - quarter, season
7. Rolling statistics - based on time windows
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from typing import Dict, List, Optional, Union, Any
import holidays

logger = logging.getLogger(__name__)


class DateTimeProcessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive datetime processor for fraud detection.
    
    This transformer extracts various temporal features from datetime columns
    to capture patterns in transaction timing.
    
    Attributes:
        datetime_columns (list): List of datetime columns to process
        features_to_extract (list): List of features to extract
        reference_date (datetime): Reference date for relative calculations
        country_holidays (str): Country code for holiday detection
    """
    
    def __init__(
        self,
        datetime_columns: Optional[List[str]] = None,
        extract_features: Optional[List[str]] = None,
        add_cyclical_features: bool = True,
        add_time_differences: bool = True,
        add_business_hours: bool = True,
        add_holidays: bool = True,
        add_seasons: bool = True,
        reference_date: Optional[datetime] = None,
        country_code: str = 'US',
        **kwargs
    ):
        """
        Initialize the DateTimeProcessor.
        
        Args:
            datetime_columns: List of datetime columns to process
            extract_features: List of features to extract
            add_cyclical_features: Whether to add sin/cos transformations
            add_time_differences: Whether to add time difference features
            add_business_hours: Whether to add business hours flags
            add_holidays: Whether to add holiday flags
            add_seasons: Whether to add seasonal features
            reference_date: Reference date for relative calculations
            country_code: Country code for holiday detection
            **kwargs: Additional arguments
        """
        self.datetime_columns = datetime_columns or []
        self.extract_features = extract_features or [
            'year', 'month', 'day', 'hour', 'minute', 'second',
            'dayofweek', 'quarter', 'dayofyear', 'weekofyear'
        ]
        self.add_cyclical_features = add_cyclical_features
        self.add_time_differences = add_time_differences
        self.add_business_hours = add_business_hours
        self.add_holidays = add_holidays
        self.add_seasons = add_seasons
        self.reference_date = reference_date or datetime.now()
        self.country_code = country_code
        self.kwargs = kwargs
        
        # Storage for fitted data
        self.fitted_columns = []
        self.holiday_calendar = None
        self.feature_names_ = []
        
        logger.info(f"DateTimeProcessor initialized for columns: {datetime_columns}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the processor to the data.
        
        Args:
            X: Input DataFrame
            y: Ignored
            
        Returns:
            self
        """
        logger.info(f"Fitting DateTimeProcessor on {len(X)} samples")
        
        # Auto-detect datetime columns if not specified
        if not self.datetime_columns:
            self.datetime_columns = self._detect_datetime_columns(X)
            logger.info(f"Auto-detected datetime columns: {self.datetime_columns}")
        
        # Initialize holiday calendar if needed
        if self.add_holidays:
            try:
                self.holiday_calendar = holidays.country_holidays(self.country_code)
                logger.info(f"Loaded holiday calendar for {self.country_code}")
            except Exception as e:
                logger.warning(f"Could not load holiday calendar: {e}")
                self.add_holidays = False
        
        # Store column names that will be created
        self._generate_feature_names(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform datetime columns to extract features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with added datetime features
        """
        logger.info(f"Transforming {len(X)} samples")
        
        result = X.copy()
        
        for col in self.datetime_columns:
            if col not in result.columns:
                logger.warning(f"Datetime column {col} not found")
                continue
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(result[col]):
                result[col] = pd.to_datetime(result[col], errors='coerce')
            
            # Extract basic datetime components
            result = self._extract_basic_features(result, col)
            
            # Add cyclical features
            if self.add_cyclical_features:
                result = self._add_cyclical_features(result, col)
            
            # Add time difference features (requires multiple datetime columns)
            if self.add_time_differences and len(self.datetime_columns) > 1:
                result = self._add_time_differences(result, col)
            
            # Add business hours features
            if self.add_business_hours:
                result = self._add_business_hours(result, col)
            
            # Add holiday features
            if self.add_holidays and self.holiday_calendar is not None:
                result = self._add_holiday_features(result, col)
            
            # Add seasonal features
            if self.add_seasons:
                result = self._add_seasonal_features(result, col)
        
        logger.info(f"Added {len(self.feature_names_)} datetime features")
        return result
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        
        Args:
            X: Input DataFrame
            y: Ignored
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _detect_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically detect datetime columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of datetime column names
        """
        datetime_cols = []
        
        for col in df.columns:
            # Check if already datetime type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
                continue
            
            # Try to convert a sample to datetime
            try:
                sample = df[col].dropna().iloc[0] if len(df) > 0 else None
                if sample is not None:
                    pd.to_datetime(sample)
                    # If successful, check more samples
                    test_samples = df[col].dropna().head(100)
                    success_rate = sum(pd.to_datetime(test_samples, errors='coerce').notna()) / len(test_samples)
                    if success_rate > 0.9:  # 90% success rate
                        datetime_cols.append(col)
            except:
                continue
        
        return datetime_cols
    
    def _extract_basic_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Extract basic datetime components.
        
        Args:
            df: Input DataFrame
            col: Datetime column name
            
        Returns:
            DataFrame with extracted features
        """
        dt_series = df[col]
        
        feature_mapping = {
            'year': dt_series.dt.year,
            'month': dt_series.dt.month,
            'day': dt_series.dt.day,
            'hour': dt_series.dt.hour,
            'minute': dt_series.dt.minute,
            'second': dt_series.dt.second,
            'dayofweek': dt_series.dt.dayofweek,
            'quarter': dt_series.dt.quarter,
            'dayofyear': dt_series.dt.dayofyear,
            'weekofyear': dt_series.dt.isocalendar().week.astype('Int64'),
            'is_month_start': dt_series.dt.is_month_start.astype(int),
            'is_month_end': dt_series.dt.is_month_end.astype(int),
            'is_quarter_start': dt_series.dt.is_quarter_start.astype(int),
            'is_quarter_end': dt_series.dt.is_quarter_end.astype(int),
            'is_year_start': dt_series.dt.is_year_start.astype(int),
            'is_year_end': dt_series.dt.is_year_end.astype(int)
        }
        
        for feature_name, feature_values in feature_mapping.items():
            if feature_name in self.extract_features:
                df[f"{col}_{feature_name}"] = feature_values
        
        return df
    
    def _add_cyclical_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Add cyclical encoding for periodic features.
        
        Cyclical encoding uses sin/cos transformations to preserve
        the cyclical nature of time features (e.g., hour 23 and hour 0
        are close in time, but numerically far apart).
        
        Args:
            df: Input DataFrame
            col: Datetime column name
            
        Returns:
            DataFrame with cyclical features
        """
        # Hour of day (0-23)
        if 'hour' in self.extract_features:
            hour = df[f"{col}_hour"] if f"{col}_hour" in df.columns else df[col].dt.hour
            df[f"{col}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df[f"{col}_hour_cos"] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0-6, Monday=0)
        if 'dayofweek' in self.extract_features:
            dow = df[f"{col}_dayofweek"] if f"{col}_dayofweek" in df.columns else df[col].dt.dayofweek
            df[f"{col}_dow_sin"] = np.sin(2 * np.pi * dow / 7)
            df[f"{col}_dow_cos"] = np.cos(2 * np.pi * dow / 7)
        
        # Month (1-12)
        if 'month' in self.extract_features:
            month = df[f"{col}_month"] if f"{col}_month" in df.columns else df[col].dt.month
            df[f"{col}_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
            df[f"{col}_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
        
        # Day of month (1-31)
        if 'day' in self.extract_features:
            day = df[f"{col}_day"] if f"{col}_day" in df.columns else df[col].dt.day
            df[f"{col}_day_sin"] = np.sin(2 * np.pi * day / 31)
            df[f"{col}_day_cos"] = np.cos(2 * np.pi * day / 31)
        
        return df
    
    def _add_time_differences(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Add time difference features between datetime columns.
        
        Args:
            df: Input DataFrame
            col: Current datetime column
            
        Returns:
            DataFrame with time difference features
        """
        # Compare with other datetime columns
        for other_col in self.datetime_columns:
            if other_col != col and other_col in df.columns:
                # Ensure both are datetime
                if pd.api.types.is_datetime64_any_dtype(df[other_col]):
                    # Calculate time difference in seconds
                    time_diff = (df[col] - df[other_col]).dt.total_seconds()
                    
                    # Add multiple representations
                    df[f"{col}_since_{other_col}_sec"] = time_diff
                    df[f"{col}_since_{other_col}_min"] = time_diff / 60
                    df[f"{col}_since_{other_col}_hour"] = time_diff / 3600
                    df[f"{col}_since_{other_col}_day"] = time_diff / 86400
                    
                    # Add absolute difference
                    df[f"{col}_abs_diff_{other_col}_hour"] = abs(time_diff) / 3600
        
        return df
    
    def _add_business_hours(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Add business hours related features.
        
        Args:
            df: Input DataFrame
            col: Datetime column name
            
        Returns:
            DataFrame with business hours features
        """
        hour = df[f"{col}_hour"] if f"{col}_hour" in df.columns else df[col].dt.hour
        dow = df[f"{col}_dayofweek"] if f"{col}_dayofweek" in df.columns else df[col].dt.dayofweek
        
        # Define business hours (9 AM - 5 PM, Monday-Friday)
        is_business_hour = (dow < 5) & (hour >= 9) & (hour < 17)
        df[f"{col}_is_business_hour"] = is_business_hour.astype(int)
        
        # Define banking hours (extended: 8 AM - 6 PM)
        is_banking_hour = (dow < 5) & (hour >= 8) & (hour < 18)
        df[f"{col}_is_banking_hour"] = is_banking_hour.astype(int)
        
        # Weekend flag
        df[f"{col}_is_weekend"] = (dow >= 5).astype(int)
        
        # Early morning (midnight - 6 AM) - suspicious time
        df[f"{col}_is_early_morning"] = ((hour >= 0) & (hour < 6)).astype(int)
        
        # Late night (10 PM - midnight) - suspicious time
        df[f"{col}_is_late_night"] = ((hour >= 22) | (hour < 4)).astype(int)
        
        return df
    
    def _add_holiday_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Add holiday detection features.
        
        Args:
            df: Input DataFrame
            col: Datetime column name
            
        Returns:
            DataFrame with holiday features
        """
        # Check if date is a holiday
        is_holiday = df[col].dt.date.apply(
            lambda x: x in self.holiday_calendar if pd.notna(x) else False
        )
        df[f"{col}_is_holiday"] = is_holiday.astype(int)
        
        # Check if it's a day before holiday
        day_before = df[col].dt.date.apply(
            lambda x: (x + timedelta(days=1)) in self.holiday_calendar if pd.notna(x) else False
        )
        df[f"{col}_is_day_before_holiday"] = day_before.astype(int)
        
        # Check if it's a day after holiday
        day_after = df[col].dt.date.apply(
            lambda x: (x - timedelta(days=1)) in self.holiday_calendar if pd.notna(x) else False
        )
        df[f"{col}_is_day_after_holiday"] = day_after.astype(int)
        
        # Weekend holiday (holiday falling on weekend)
        is_weekend_holiday = (df[f"{col}_is_weekend"] == 1) & (df[f"{col}_is_holiday"] == 1)
        df[f"{col}_is_weekend_holiday"] = is_weekend_holiday.astype(int)
        
        return df
    
    def _add_seasonal_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Add seasonal features based on month.
        
        Args:
            df: Input DataFrame
            col: Datetime column name
            
        Returns:
            DataFrame with seasonal features
        """
        month = df[f"{col}_month"] if f"{col}_month" in df.columns else df[col].dt.month
        
        # Define seasons (Northern Hemisphere)
        season_map = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        }
        
        df[f"{col}_season"] = month.map(season_map)
        
        # One-hot encode seasons (or create flags)
        seasons = ['winter', 'spring', 'summer', 'fall']
        for season in seasons:
            df[f"{col}_is_{season}"] = (df[f"{col}_season"] == season).astype(int)
        
        # Holiday season (November-December)
        df[f"{col}_is_holiday_season"] = month.isin([11, 12]).astype(int)
        
        # Summer vacation months
        df[f"{col}_is_summer_vacation"] = month.isin([6, 7, 8]).astype(int)
        
        return df
    
    def _generate_feature_names(self, X: pd.DataFrame):
        """
        Generate names of features that will be created.
        
        Args:
            X: Input DataFrame
        """
        self.feature_names_ = []
        
        for col in self.datetime_columns:
            # Basic features
            for feature in self.extract_features:
                self.feature_names_.append(f"{col}_{feature}")
            
            # Cyclical features
            if self.add_cyclical_features:
                if 'hour' in self.extract_features:
                    self.feature_names_.extend([f"{col}_hour_sin", f"{col}_hour_cos"])
                if 'dayofweek' in self.extract_features:
                    self.feature_names_.extend([f"{col}_dow_sin", f"{col}_dow_cos"])
                if 'month' in self.extract_features:
                    self.feature_names_.extend([f"{col}_month_sin", f"{col}_month_cos"])
                if 'day' in self.extract_features:
                    self.feature_names_.extend([f"{col}_day_sin", f"{col}_day_cos"])
            
            # Business hours features
            if self.add_business_hours:
                self.feature_names_.extend([
                    f"{col}_is_business_hour",
                    f"{col}_is_banking_hour",
                    f"{col}_is_weekend",
                    f"{col}_is_early_morning",
                    f"{col}_is_late_night"
                ])
            
            # Holiday features
            if self.add_holidays:
                self.feature_names_.extend([
                    f"{col}_is_holiday",
                    f"{col}_is_day_before_holiday",
                    f"{col}_is_day_after_holiday",
                    f"{col}_is_weekend_holiday"
                ])
            
            # Seasonal features
            if self.add_seasons:
                self.feature_names_.extend([
                    f"{col}_is_winter",
                    f"{col}_is_spring",
                    f"{col}_is_summer",
                    f"{col}_is_fall",
                    f"{col}_is_holiday_season",
                    f"{col}_is_summer_vacation"
                ])
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Get output feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names_
    
    def get_processor_info(self) -> Dict:
        """
        Get information about datetime processing.
        
        Returns:
            Dictionary with processing information
        """
        return {
            'datetime_columns': self.datetime_columns,
            'extracted_features': self.extract_features,
            'cyclical_features': self.add_cyclical_features,
            'business_hours': self.add_business_hours,
            'holidays': self.add_holidays,
            'seasons': self.add_seasons,
            'feature_count': len(self.feature_names_)
        }