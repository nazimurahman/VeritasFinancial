"""
Seasonality Features for Banking Fraud Detection
================================================
This module implements features that capture cyclical patterns in transaction
behavior. Seasonality is crucial for distinguishing normal spending patterns
from fraudulent activities.

Key Concepts:
- Time-based cycles: Hour of day, day of week, month, quarter
- Cyclical encoding: Sin/cos transformations to preserve circular relationships
- Holiday effects: Special day patterns
- Payday cycles: Monthly income patterns
- Weekend vs weekday behavior
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import holidays


class SeasonalityFeatureEngineer:
    """
    Create features capturing cyclical patterns in transaction data.
    
    Banking Context:
    - Most people have regular spending patterns (groceries on weekends,
      bills on weekdays, etc.)
    - Fraud often occurs at unusual times (3 AM, holidays)
    - Seasonal fraud patterns (holiday shopping fraud, tax season scams)
    """
    
    def __init__(self, country_code: str = 'US'):
        """
        Initialize seasonality engineer.
        
        Parameters:
        -----------
        country_code : ISO country code for holiday calendar
        """
        self.country_code = country_code
        self.feature_columns = []
        
        # Initialize holiday calendar
        try:
            self.holiday_calendar = holidays.country_holidays(country_code)
        except:
            # Fallback to empty if country not supported
            self.holiday_calendar = {}
            print(f"Warning: Holiday calendar not available for {country_code}")
    
    def create_cyclical_time_features(self, 
                                      df: pd.DataFrame,
                                      time_col: str = 'transaction_time') -> pd.DataFrame:
        """
        Create cyclical encoding of time features.
        
        Why cyclical encoding?
        - Time features are circular: 23:00 and 01:00 are close in reality
        - Simple one-hot encoding loses this relationship
        - Sin/cos transformations preserve circular continuity
        
        Parameters:
        -----------
        df : DataFrame with timestamp column
        time_col : Name of timestamp column
        
        Returns:
        --------
        DataFrame with sin/cos encoded time features
        """
        
        result_df = df.copy()
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(result_df[time_col]):
            result_df[time_col] = pd.to_datetime(result_df[time_col])
        
        # Extract time components
        result_df['hour'] = result_df[time_col].dt.hour
        result_df['minute'] = result_df[time_col].dt.minute
        result_df['day_of_week'] = result_df[time_col].dt.dayofweek  # 0=Monday, 6=Sunday
        result_df['day_of_month'] = result_df[time_col].dt.day
        result_df['week_of_year'] = result_df[time_col].dt.isocalendar().week.astype(int)
        result_df['month'] = result_df[time_col].dt.month
        result_df['quarter'] = result_df[time_col].dt.quarter
        result_df['year'] = result_df[time_col].dt.year
        result_df['day_of_year'] = result_df[time_col].dt.dayofyear
        
        # Cyclical encoding for hour (24-hour cycle)
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
        self.feature_columns.extend(['hour_sin', 'hour_cos'])
        
        # Cyclical encoding for minute (60-minute cycle)
        result_df['minute_sin'] = np.sin(2 * np.pi * result_df['minute'] / 60)
        result_df['minute_cos'] = np.cos(2 * np.pi * result_df['minute'] / 60)
        self.feature_columns.extend(['minute_sin', 'minute_cos'])
        
        # Cyclical encoding for day of week (7-day cycle)
        result_df['dow_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['dow_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        self.feature_columns.extend(['dow_sin', 'dow_cos'])
        
        # Cyclical encoding for month (12-month cycle)
        result_df['month_sin'] = np.sin(2 * np.pi * (result_df['month'] - 1) / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * (result_df['month'] - 1) / 12)
        self.feature_columns.extend(['month_sin', 'month_cos'])
        
        # Cyclical encoding for day of month (approximately 30-day cycle)
        result_df['dom_sin'] = np.sin(2 * np.pi * result_df['day_of_month'] / 31)
        result_df['dom_cos'] = np.cos(2 * np.pi * result_df['day_of_month'] / 31)
        self.feature_columns.extend(['dom_sin', 'dom_cos'])
        
        return result_df
    
    def create_business_cycle_features(self,
                                      df: pd.DataFrame,
                                      time_col: str = 'transaction_time') -> pd.DataFrame:
        """
        Create features related to business cycles and working hours.
        
        Banking Context:
        - Fraud often occurs outside normal business hours
        - Different patterns for weekends vs weekdays
        - Rush hour transactions may have different risk profiles
        """
        
        result_df = df.copy()
        
        # Ensure time features exist
        if 'hour' not in result_df.columns:
            result_df = self.create_cyclical_time_features(result_df, time_col)
        
        # Business hours (9 AM - 5 PM on weekdays)
        result_df['is_business_hours'] = (
            (result_df['hour'] >= 9) & 
            (result_df['hour'] <= 17) & 
            (result_df['day_of_week'] < 5)  # Monday=0 to Friday=4
        ).astype(int)
        self.feature_columns.append('is_business_hours')
        
        # Late night/early morning (high-risk period for fraud)
        result_df['is_late_night'] = (
            (result_df['hour'] >= 0) & 
            (result_df['hour'] <= 5)
        ).astype(int)
        self.feature_columns.append('is_late_night')
        
        # Weekend indicator
        result_df['is_weekend'] = (result_df['day_of_week'] >= 5).astype(int)  # 5=Saturday, 6=Sunday
        self.feature_columns.append('is_weekend')
        
        # Rush hours (morning and evening commutes)
        result_df['is_morning_rush'] = (
            (result_df['hour'] >= 7) & 
            (result_df['hour'] <= 9) & 
            (result_df['day_of_week'] < 5)
        ).astype(int)
        
        result_df['is_evening_rush'] = (
            (result_df['hour'] >= 16) & 
            (result_df['hour'] <= 19) & 
            (result_df['day_of_week'] < 5)
        ).astype(int)
        
        result_df['is_rush_hour'] = (
            result_df['is_morning_rush'] | result_df['is_evening_rush']
        ).astype(int)
        
        self.feature_columns.extend(['is_morning_rush', 'is_evening_rush', 'is_rush_hour'])
        
        # Quarter end (often higher legitimate spending)
        result_df['is_quarter_end'] = (result_df['month'].isin([3, 6, 9, 12])).astype(int)
        self.feature_columns.append('is_quarter_end')
        
        # Month end (payday effect for many)
        result_df['is_month_end'] = (result_df['day_of_month'] >= 25).astype(int)
        self.feature_columns.append('is_month_end')
        
        return result_df
    
    def create_holiday_features(self,
                               df: pd.DataFrame,
                               time_col: str = 'transaction_time',
                               lookback_days: int = 3,
                               lookahead_days: int = 3) -> pd.DataFrame:
        """
        Create features related to holidays.
        
        Holidays are special periods with:
        - Increased shopping activity
        - Different fraud patterns
        - Potential system delays
        
        Parameters:
        -----------
        lookback_days : Days before holiday to consider as "holiday period"
        lookahead_days : Days after holiday to consider
        """
        
        result_df = df.copy()
        
        # Extract date (without time)
        result_df['date'] = result_df[time_col].dt.date
        
        # Check if date is a holiday
        result_df['is_holiday'] = result_df['date'].apply(
            lambda x: x in self.holiday_calendar
        ).astype(int)
        self.feature_columns.append('is_holiday')
        
        # Holiday name if available
        result_df['holiday_name'] = result_df['date'].apply(
            lambda x: self.holiday_calendar.get(x, '')
        )
        
        # Days to next holiday
        holidays_dates = sorted([date for date in self.holiday_calendar.keys() 
                                 if isinstance(date, (datetime, pd.Timestamp))])
        
        if holidays_dates:
            # Convert to datetime for comparison
            holidays_dates = [pd.to_datetime(date) for date in holidays_dates]
            
            # Find next holiday
            def days_to_next_holiday(current_date):
                future_holidays = [h for h in holidays_dates if h >= pd.to_datetime(current_date)]
                if future_holidays:
                    return (future_holidays[0] - pd.to_datetime(current_date)).days
                return 365  # Large number if no future holiday
            
            result_df['days_to_next_holiday'] = result_df['date'].apply(days_to_next_holiday)
            
            # Days since last holiday
            def days_since_last_holiday(current_date):
                past_holidays = [h for h in holidays_dates if h <= pd.to_datetime(current_date)]
                if past_holidays:
                    return (pd.to_datetime(current_date) - past_holidays[-1]).days
                return 365  # Large number if no past holiday
            
            result_df['days_since_last_holiday'] = result_df['date'].apply(days_since_last_holiday)
            
            # Holiday period indicators
            result_df['is_pre_holiday'] = (
                (result_df['days_to_next_holiday'] <= lookback_days) & 
                (result_df['days_to_next_holiday'] > 0)
            ).astype(int)
            
            result_df['is_post_holiday'] = (
                (result_df['days_since_last_holiday'] <= lookahead_days) & 
                (result_df['days_since_last_holiday'] > 0)
            ).astype(int)
            
            result_df['is_holiday_period'] = (
                result_df['is_pre_holiday'] | 
                result_df['is_holiday'] | 
                result_df['is_post_holiday']
            ).astype(int)
            
            self.feature_columns.extend([
                'days_to_next_holiday', 
                'days_since_last_holiday',
                'is_pre_holiday',
                'is_post_holiday',
                'is_holiday_period'
            ])
        
        return result_df
    
    def create_payday_cycle_features(self,
                                    df: pd.DataFrame,
                                    time_col: str = 'transaction_time',
                                    payday_days: List[int] = [1, 15, 25, 30]) -> pd.DataFrame:
        """
        Create features related to payday cycles.
        
        Many fraudsters exploit payday patterns:
        - Increased account balances after payday
        - Higher spending immediately after payday
        - Bills typically paid around certain dates
        
        Parameters:
        -----------
        payday_days : List of days of month considered as paydays
        """
        
        result_df = df.copy()
        
        if 'day_of_month' not in result_df.columns:
            result_df['day_of_month'] = result_df[time_col].dt.day
        
        # Is this transaction on a payday?
        result_df['is_payday'] = result_df['day_of_month'].isin(payday_days).astype(int)
        self.feature_columns.append('is_payday')
        
        # Days since last payday
        def days_since_payday(day):
            # Find the most recent payday
            past_paydays = [d for d in payday_days if d <= day]
            if past_paydays:
                return day - max(past_paydays)
            else:
                # If no payday this month, go to previous month
                last_month_payday = max(payday_days)
                return day + (30 - last_month_payday)  # Approximate
        
        result_df['days_since_payday'] = result_df['day_of_month'].apply(days_since_payday)
        
        # Days to next payday
        def days_to_payday(day):
            future_paydays = [d for d in payday_days if d >= day]
            if future_paydays:
                return min(future_paydays) - day
            else:
                # If no payday this month, go to next month
                next_month_payday = min(payday_days)
                return (30 - day) + next_month_payday
        
        result_df['days_to_payday'] = result_df['day_of_month'].apply(days_to_payday)
        
        # Payday week (3 days before to 3 days after payday)
        result_df['is_payday_week'] = (
            (result_df['days_to_payday'] <= 3) | 
            (result_df['days_since_payday'] <= 3)
        ).astype(int)
        
        self.feature_columns.extend([
            'days_since_payday',
            'days_to_payday',
            'is_payday_week'
        ])
        
        return result_df
    
    def create_seasonal_pattern_features(self,
                                        df: pd.DataFrame,
                                        customer_id_col: str = 'customer_id',
                                        amount_col: str = 'amount',
                                        time_col: str = 'transaction_time') -> pd.DataFrame:
        """
        Create features capturing customer-specific seasonal patterns.
        
        This identifies deviations from a customer's typical seasonal behavior.
        For example, if a customer usually shops on weekends but suddenly
        starts shopping on weekdays, it might indicate account takeover.
        """
        
        result_df = df.copy()
        
        # Ensure we have time components
        if 'hour' not in result_df.columns:
            result_df = self.create_cyclical_time_features(result_df, time_col)
        
        # Calculate customer's typical patterns
        # Typical hour of day
        customer_hour_avg = result_df.groupby(customer_id_col)['hour'].mean()
        result_df['customer_avg_hour'] = result_df[customer_id_col].map(customer_hour_avg)
        
        # Deviation from typical hour
        result_df['hour_deviation'] = abs(result_df['hour'] - result_df['customer_avg_hour'])
        result_df['hour_deviation_normalized'] = (
            result_df['hour_deviation'] / 12  # Normalize to 0-1 range
        ).clip(0, 1)
        
        # Typical day of week
        customer_dow_avg = result_df.groupby(customer_id_col)['day_of_week'].mean()
        result_df['customer_avg_dow'] = result_df[customer_id_col].map(customer_dow_avg)
        result_df['dow_deviation'] = abs(result_df['day_of_week'] - result_df['customer_avg_dow'])
        result_df['dow_deviation_normalized'] = (
            result_df['dow_deviation'] / 3.5  # Normalize (max deviation ~3.5)
        ).clip(0, 1)
        
        # Weekend vs weekday preference
        customer_weekend_ratio = (
            result_df.groupby(customer_id_col)['is_weekend'].mean()
        )
        result_df['customer_weekend_ratio'] = result_df[customer_id_col].map(customer_weekend_ratio)
        result_df['weekend_deviation'] = abs(
            result_df['is_weekend'] - result_df['customer_weekend_ratio']
        )
        
        # Seasonal spending patterns
        # Amount by month (seasonal spending)
        monthly_spending = result_df.groupby([customer_id_col, 'month'])[amount_col].mean()
        monthly_spending.name = 'customer_monthly_avg'
        result_df = result_df.merge(
            monthly_spending.reset_index(),
            on=[customer_id_col, 'month'],
            how='left'
        )
        
        # Deviation from seasonal pattern
        result_df['seasonal_amount_ratio'] = (
            result_df[amount_col] / (result_df['customer_monthly_avg'] + 1e-8)
        )
        
        self.feature_columns.extend([
            'hour_deviation',
            'hour_deviation_normalized',
            'dow_deviation',
            'dow_deviation_normalized',
            'weekend_deviation',
            'customer_weekend_ratio',
            'customer_monthly_avg',
            'seasonal_amount_ratio'
        ])
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Return list of created feature names."""
        return self.feature_columns