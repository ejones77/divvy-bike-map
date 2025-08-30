import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Comprehensive feature engineering for bike availability prediction."""
    
    @staticmethod
    def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create all features - let feature selection decide what's useful."""
        df = df.copy()
        
        # Time features
        df['hour_of_day'] = df['recorded_at'].dt.hour
        df['day_of_week'] = df['recorded_at'].dt.dayofweek + 1
        df['is_weekend'] = df['day_of_week'].isin([6, 7]).astype(int)
        df['month'] = df['recorded_at'].dt.month
        df['day_of_month'] = df['recorded_at'].dt.day
        
        # Peak hour indicators
        df['is_morning_peak'] = df['hour_of_day'].isin([7, 8, 9]).astype(int)
        df['is_evening_peak'] = df['hour_of_day'].isin([17, 18, 19]).astype(int)
        df['is_peak_hours'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Historical lag features
        df['num_bikes_1h_ago'] = df.groupby('station_id')['num_bikes_available'].shift(4).fillna(df['num_bikes_available'])
        df['num_bikes_3h_ago'] = df.groupby('station_id')['num_bikes_available'].shift(12).fillna(df['num_bikes_available'])
        
        # Rate of change features
        df['bikes_change_1h'] = df['num_bikes_available'] - df['num_bikes_1h_ago']
        df['bikes_change_3h'] = df['num_bikes_available'] - df['num_bikes_3h_ago']
        df['bikes_pct_change_1h'] = df['bikes_change_1h'] / (df['num_bikes_1h_ago'] + 1)
        
        # Rolling features (multiple windows)
        for window in [8, 24, 48]:  # 2h, 6h, 12h
            window_name = f"{window//4}h"
            df[f'avg_bikes_{window_name}'] = df.groupby('station_id')['num_bikes_available'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'std_bikes_{window_name}'] = df.groupby('station_id')['num_bikes_available'].transform(
                lambda x: x.rolling(window, min_periods=2).std().fillna(0)
            )
            df[f'min_bikes_{window_name}'] = df.groupby('station_id')['num_bikes_available'].transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
            df[f'max_bikes_{window_name}'] = df.groupby('station_id')['num_bikes_available'].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
        
        # Trend features
        df['trend_last_2h'] = df['num_bikes_available'] - df['avg_bikes_2h']
        df['bikes_vs_avg_ratio'] = df['num_bikes_available'] / (df['avg_bikes_2h'] + 1)
        df['trend_acceleration'] = df['trend_last_2h'] - df.groupby('station_id')['trend_last_2h'].shift(1)
        df['volatility_2h'] = df['std_bikes_2h']
        
        # Capacity and utilization features
        df['capacity_utilization'] = df['num_bikes_available'] / df['capacity']
        df['capacity_pressure'] = (df['capacity'] - df['num_bikes_available']) / df['capacity']
        df['bikes_vs_docks_ratio'] = df['num_bikes_available'] / (df['num_docks_available'] + 1)
        df['capacity_remaining'] = df['capacity'] - df['num_bikes_available']
        
        # Station operational features
        df['station_reliability'] = (df['is_installed'] & df['is_renting'] & df['is_returning']).astype(int)
        df['operational_score'] = df['is_installed'] + df['is_renting'] + df['is_returning']
        
        # Polynomial features for key ratios
        df['availability_ratio_squared'] = df['availability_ratio'] ** 2
        df['availability_ratio_cubed'] = df['availability_ratio'] ** 3
        df['availability_ratio_log'] = np.log1p(df['availability_ratio'])
        df['availability_ratio_sqrt'] = np.sqrt(df['availability_ratio'])
        
        # Interaction features
        df['hour_dow_interaction'] = df['hour_of_day'] * df['day_of_week']
        df['hour_weekend_interaction'] = df['hour_of_day'] * df['is_weekend']
        df['capacity_hour_interaction'] = df['capacity'] * df['hour_of_day']
        df['weekend_capacity_interaction'] = df['is_weekend'] * df['capacity']
        
        # Spatial interactions
        if 'lat' in df.columns and 'lon' in df.columns:
            df['lat_lon_interaction'] = df['lat'] * df['lon']
        
        # Categorical interaction (done last)
        df['dow_hour_interaction'] = df['day_of_week'].astype(str) + '_' + df['hour_of_day'].astype(str)
        
        logger.info(f"feature_engineering_complete shape={df.shape}")
        return df
