"""
Merchant Aggregate Features for Banking Fraud Detection
=======================================================
This module implements features aggregated at the merchant level.
Merchant risk profiling is crucial for detecting fraud patterns
associated with specific merchants or merchant categories.

Key Concepts:
- Merchant risk scoring: Historical fraud rates by merchant
- Merchant category analysis: Risk by business type
- Merchant geography: Location-based risk
- Merchant behavior patterns: Transaction patterns by merchant
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


class MerchantAggregateFeatureEngineer:
    """
    Create merchant-level aggregate features.
    
    Banking Context:
    - Some merchants have higher fraud rates (e.g., electronics, gift cards)
    - New merchants may be higher risk
    - Merchant location can indicate risk (high-fraud regions)
    - First-time merchant for customer is riskier
    """
    
    def __init__(self):
        self.feature_columns = []
        self.merchant_profiles = {}
        
    def create_merchant_risk_features(self,
                                     df: pd.DataFrame,
                                     merchant_id_col: str = 'merchant_id',
                                     fraud_col: str = 'is_fraud',
                                     amount_col: str = 'amount') -> pd.DataFrame:
        """
        Create merchant risk features based on historical fraud rates.
        
        Parameters:
        -----------
        df : DataFrame with transaction data
        merchant_id_col : Merchant identifier
        fraud_col : Fraud label column
        amount_col : Transaction amount column
        
        Returns:
        --------
        DataFrame with merchant risk features
        """
        
        result_df = df.copy()
        
        # Calculate merchant-level statistics
        merchant_stats = result_df.groupby(merchant_id_col).agg({
            fraud_col: [
                ('merchant_fraud_count', 'sum'),
                ('merchant_fraud_rate', 'mean'),
            ],
            'transaction_id': [
                ('merchant_tx_count', 'count')
            ],
            amount_col: [
                ('merchant_avg_amount', 'mean'),
                ('merchant_std_amount', 'std'),
                ('merchant_total_volume', 'sum'),
                ('merchant_max_amount', 'max'),
                ('merchant_min_amount', 'min')
            ]
        })
        
        # Flatten multi-level columns
        merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns.values]
        merchant_stats = merchant_stats.reset_index()
        
        # Add derived risk metrics
        # Fraud severity (average amount of fraudulent transactions)
        fraud_amounts = result_df[result_df[fraud_col] == 1].groupby(merchant_id_col)[amount_col].mean()
        fraud_amounts.name = 'merchant_avg_fraud_amount'
        merchant_stats = merchant_stats.merge(
            fraud_amounts.reset_index(),
            on=merchant_id_col,
            how='left'
        )
        merchant_stats['merchant_avg_fraud_amount'] = merchant_stats['merchant_avg_fraud_amount'].fillna(0)
        
        # Fraud ratio relative to overall fraud rate
        overall_fraud_rate = result_df[fraud_col].mean()
        merchant_stats['merchant_fraud_rate_ratio'] = (
            merchant_stats['merchant_fraud_rate'] / (overall_fraud_rate + 1e-8)
        )
        
        # Risk score based on fraud rate and volume
        # Using Bayesian smoothing to handle low-volume merchants
        global_avg_fraud = overall_fraud_rate
        confidence_weight = 30  # Equivalent number of transactions for smoothing
        
        merchant_stats['merchant_risk_score'] = (
            (merchant_stats['merchant_fraud_count'] + confidence_weight * global_avg_fraud) /
            (merchant_stats['merchant_tx_count'] + confidence_weight)
        )
        
        # Risk category
        merchant_stats['merchant_risk_category'] = pd.cut(
            merchant_stats['merchant_risk_score'],
            bins=[0, 0.01, 0.05, 0.1, 1.0],
            labels=['very_low', 'low', 'medium', 'high']
        )
        
        # Merge back to original dataframe
        result_df = result_df.merge(
            merchant_stats[[
                merchant_id_col, 
                'merchant_fraud_rate',
                'merchant_tx_count',
                'merchant_avg_amount',
                'merchant_std_amount',
                'merchant_total_volume',
                'merchant_max_amount',
                'merchant_risk_score',
                'merchant_risk_category',
                'merchant_avg_fraud_amount',
                'merchant_fraud_rate_ratio'
            ]],
            on=merchant_id_col,
            how='left'
        )
        
        # Fill NaN for merchants with no history
        fill_values = {
            'merchant_fraud_rate': overall_fraud_rate,
            'merchant_tx_count': 0,
            'merchant_avg_amount': result_df[amount_col].mean(),
            'merchant_std_amount': 0,
            'merchant_total_volume': 0,
            'merchant_max_amount': result_df[amount_col].max(),
            'merchant_risk_score': global_avg_fraud,
            'merchant_avg_fraud_amount': 0,
            'merchant_fraud_rate_ratio': 1.0
        }
        result_df = result_df.fillna(fill_values)
        
        # Create risk flags
        result_df['is_high_risk_merchant'] = (
            result_df['merchant_risk_score'] > 0.1
        ).astype(int)
        
        result_df['is_medium_risk_merchant'] = (
            (result_df['merchant_risk_score'] > 0.05) & 
            (result_df['merchant_risk_score'] <= 0.1)
        ).astype(int)
        
        result_df['is_low_volume_merchant'] = (
            result_df['merchant_tx_count'] < 10
        ).astype(int)
        
        self.feature_columns.extend([
            'merchant_fraud_rate',
            'merchant_tx_count',
            'merchant_avg_amount',
            'merchant_std_amount',
            'merchant_total_volume',
            'merchant_max_amount',
            'merchant_risk_score',
            'merchant_avg_fraud_amount',
            'merchant_fraud_rate_ratio',
            'is_high_risk_merchant',
            'is_medium_risk_merchant',
            'is_low_volume_merchant'
        ])
        
        # Store merchant profiles
        self.merchant_profiles = merchant_stats.set_index(merchant_id_col).to_dict(orient='index')
        
        return result_df
    
    def create_merchant_category_features(self,
                                         df: pd.DataFrame,
                                         merchant_category_col: str = 'merchant_category',
                                         fraud_col: str = 'is_fraud',
                                         amount_col: str = 'amount') -> pd.DataFrame:
        """
        Create features based on merchant category (MCC).
        
        Different merchant categories have different risk profiles:
        - High-risk: Gambling, adult entertainment, cryptocurrency
        - Medium-risk: Electronics, travel, jewelry
        - Low-risk: Groceries, utilities, healthcare
        """
        
        result_df = df.copy()
        
        if merchant_category_col not in result_df.columns:
            print("Warning: Merchant category column not found")
            return result_df
        
        # Category-level statistics
        category_stats = result_df.groupby(merchant_category_col).agg({
            fraud_col: [
                ('category_fraud_rate', 'mean'),
                ('category_fraud_count', 'sum')
            ],
            'transaction_id': [
                ('category_tx_count', 'count')
            ],
            amount_col: [
                ('category_avg_amount', 'mean'),
                ('category_std_amount', 'std')
            ]
        })
        
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns.values]
        category_stats = category_stats.reset_index()
        
        # Risk categories for merchant types
        # This is based on industry knowledge - adjust based on your data
        high_risk_categories = ['gambling', 'adult', 'cryptocurrency', 'money_transfer', 
                                'pawn_shop', 'casino', 'dating']
        medium_risk_categories = ['electronics', 'travel', 'jewelry', 'luxury_goods', 
                                  'online_retail', 'telemarketing']
        
        result_df['category_risk_level'] = 'low'
        result_df.loc[result_df[merchant_category_col].isin(medium_risk_categories), 'category_risk_level'] = 'medium'
        result_df.loc[result_df[merchant_category_col].isin(high_risk_categories), 'category_risk_level'] = 'high'
        
        # Encode risk level
        risk_mapping = {'low': 1, 'medium': 2, 'high': 3}
        result_df['category_risk_level_encoded'] = result_df['category_risk_level'].map(risk_mapping)
        
        # Merge category statistics
        result_df = result_df.merge(
            category_stats[[merchant_category_col, 'category_fraud_rate', 'category_tx_count']],
            on=merchant_category_col,
            how='left'
        )
        
        # Compare merchant risk to category average
        if 'merchant_risk_score' in result_df.columns:
            result_df['merchant_vs_category_risk'] = (
                result_df['merchant_risk_score'] / (result_df['category_fraud_rate'] + 1e-8)
            )
            self.feature_columns.append('merchant_vs_category_risk')
        
        self.feature_columns.extend([
            'category_risk_level_encoded',
            'category_fraud_rate',
            'category_tx_count'
        ])
        
        return result_df
    
    def create_merchant_customer_features(self,
                                         df: pd.DataFrame,
                                         customer_id_col: str = 'customer_id',
                                         merchant_id_col: str = 'merchant_id',
                                         amount_col: str = 'amount') -> pd.DataFrame:
        """
        Create features about customer-merchant relationships.
        
        These capture how familiar a customer is with a merchant,
        which is important for detecting unusual merchant choices.
        """
        
        result_df = df.copy()
        
        # Customer-merchant history
        customer_merchant_stats = result_df.groupby([customer_id_col, merchant_id_col]).agg({
            'transaction_id': [
                ('customer_merchant_tx_count', 'count')
            ],
            amount_col: [
                ('customer_merchant_avg_amount', 'mean'),
                ('customer_merchant_total_spent', 'sum'),
                ('customer_merchant_max_amount', 'max')
            ],
            'transaction_time': [
                ('customer_merchant_first_tx', 'min'),
                ('customer_merchant_last_tx', 'max')
            ]
        })
        
        customer_merchant_stats.columns = ['_'.join(col).strip() for col in customer_merchant_stats.columns.values]
        customer_merchant_stats = customer_merchant_stats.reset_index()
        
        # Merge back
        result_df = result_df.merge(
            customer_merchant_stats,
            on=[customer_id_col, merchant_id_col],
            how='left'
        )
        
        # Fill NaN for first-time customer-merchant pairs
        result_df['customer_merchant_tx_count'] = result_df['customer_merchant_tx_count'].fillna(0)
        result_df['customer_merchant_avg_amount'] = result_df['customer_merchant_avg_amount'].fillna(result_df[amount_col])
        result_df['customer_merchant_total_spent'] = result_df['customer_merchant_total_spent'].fillna(0)
        result_df['customer_merchant_max_amount'] = result_df['customer_merchant_max_amount'].fillna(0)
        
        # Create flags
        result_df['is_first_tx_with_merchant'] = (
            result_df['customer_merchant_tx_count'] == 0
        ).astype(int)
        
        result_df['is_new_merchant_for_customer'] = (
            result_df['customer_merchant_tx_count'] <= 2
        ).astype(int)
        
        # Amount compared to average with this merchant
        result_df['amount_vs_customer_merchant_avg'] = (
            result_df[amount_col] / (result_df['customer_merchant_avg_amount'] + 1e-8)
        )
        result_df['amount_vs_customer_merchant_avg'] = (
            result_df['amount_vs_customer_merchant_avg'].clip(0, 10)
        )
        
        # Is this the largest transaction with this merchant?
        result_df['is_max_with_merchant'] = (
            result_df[amount_col] >= result_df['customer_merchant_max_amount']
        ).astype(int)
        
        # Days since last transaction with this merchant
        if 'customer_merchant_last_tx' in result_df.columns:
            result_df['days_since_last_merchant_tx'] = (
                pd.Timestamp.now() - result_df['customer_merchant_last_tx']
            ).dt.total_seconds() / (24 * 3600)
            result_df['days_since_last_merchant_tx'] = result_df['days_since_last_merchant_tx'].fillna(999)
        
        self.feature_columns.extend([
            'customer_merchant_tx_count',
            'customer_merchant_avg_amount',
            'customer_merchant_total_spent',
            'is_first_tx_with_merchant',
            'is_new_merchant_for_customer',
            'amount_vs_customer_merchant_avg',
            'is_max_with_merchant'
        ])
        
        if 'days_since_last_merchant_tx' in result_df.columns:
            self.feature_columns.append('days_since_last_merchant_tx')
        
        return result_df
    
    def create_merchant_geography_features(self,
                                          df: pd.DataFrame,
                                          merchant_id_col: str = 'merchant_id',
                                          country_col: str = 'merchant_country',
                                          city_col: str = 'merchant_city') -> pd.DataFrame:
        """
        Create features based on merchant geography.
        
        Location-based risk factors:
        - Countries with high fraud rates
        - Distance from customer's location
        - Unusual geographic patterns
        """
        
        result_df = df.copy()
        
        if country_col in result_df.columns:
            # Country-level fraud statistics
            country_stats = result_df.groupby(country_col).agg({
                'is_fraud': [
                    ('country_fraud_rate', 'mean')
                ],
                'transaction_id': [
                    ('country_tx_count', 'count')
                ]
            })
            
            country_stats.columns = ['_'.join(col).strip() for col in country_stats.columns.values]
            country_stats = country_stats.reset_index()
            
            # Known high-risk countries (adjust based on your data)
            high_risk_countries = ['XX', 'YY', 'ZZ']  # Placeholder - add actual high-risk country codes
            
            result_df['is_high_risk_country'] = result_df[country_col].isin(high_risk_countries).astype(int)
            
            # Merge country statistics
            result_df = result_df.merge(
                country_stats[[country_col, 'country_fraud_rate', 'country_tx_count']],
                on=country_col,
                how='left'
            )
            
            self.feature_columns.extend([
                'is_high_risk_country',
                'country_fraud_rate',
                'country_tx_count'
            ])
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Return list of created feature names."""
        return self.feature_columns