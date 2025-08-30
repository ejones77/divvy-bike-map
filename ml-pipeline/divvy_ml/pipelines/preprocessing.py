import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from divvy_ml.utils.database import DatabaseClient
from divvy_ml.pipelines.feature_engineering import FeatureEngineering
from divvy_ml.pipelines.feature_selection import FeatureSelector, FeatureScaler, FeatureAnalyzer
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, n_features: int = 30, use_scaling: bool = True):
        logger.info(f"DataPreprocessor init starting n_features={n_features} use_scaling={use_scaling}")
        
        import os
        db_url = os.environ.get('DB_URL')
        logger.info(f"DB_URL available: {bool(db_url)}")
        
        try:
            logger.info("Attempting to initialize DatabaseClient...")
            self.db_client = DatabaseClient()
            logger.info("DatabaseClient initialized successfully")
        except Exception as e:
            logger.error(f"db_client_init_failed error={e}")
            raise RuntimeError(f"Failed to initialize database client: {e}")
            
        self.n_features = n_features
        self.use_scaling = use_scaling
        
        logger.info("Initializing feature pipeline components...")
        self.scaler = FeatureScaler()
        self.selector = FeatureSelector(n_features)
        self.analyzer = FeatureAnalyzer()
        
        self.is_fitted = False
        self.feature_columns = None
        
        logger.info("DataPreprocessor initialization completed successfully")
        
    def load_stations_data(self):
        """Load stations metadata from database."""
        try:
            if not hasattr(self, 'db_client') or self.db_client is None:
                logger.info("db_client_missing_recreating_for_inference")
                self.db_client = DatabaseClient()
            
            logger.info("loading_stations_from_db")
            return self.db_client.get_stations_metadata()
        except Exception as e:
            logger.error(f"db_stations_failed error={e}")
            raise RuntimeError(f"Failed to load stations data from database: {e}")
    
    def _load_availability_data(self, hours_back: int = 24):
        """Load availability data from database."""
        if not hasattr(self, 'db_client') or self.db_client is None:
            logger.info("db_client_missing_recreating_for_inference")
            self.db_client = DatabaseClient()
            
        return self.db_client.get_availability_data(hours_back=hours_back, inference_mode=True)
    
    def create_availability_target(self, df):
        """Create availability ratio and categorical target for 6 hours ahead."""
        df['availability_ratio'] = df['num_bikes_available'] / df['capacity'].fillna(20)
        
        # 0 = green (high >= 0.6), 1 = yellow (medium 0.3-0.6), 2 = red (low < 0.3)
        df['availability_target_current'] = df['availability_ratio'].apply(
            lambda x: 0 if x >= 0.6 else (1 if x >= 0.3 else 2)
        )
        
        prediction_horizon_hours = 6
        df['future_timestamp'] = df['recorded_at'] + pd.Timedelta(hours=prediction_horizon_hours)
        
        future_lookup = df[['station_id', 'recorded_at', 'availability_target_current']].copy()
        future_lookup = future_lookup.rename(columns={
            'recorded_at': 'future_timestamp',
            'availability_target_current': 'availability_target_future'
        })
        
        df = df.merge(
            future_lookup,
            on=['station_id', 'future_timestamp'],
            how='left'
        )
        
        df['availability_target'] = df['availability_target_future']
        
        df = df.drop(columns=['future_timestamp', 'availability_target_future'])
        
        valid_targets = df['availability_target'].notna().sum()
        total_records = len(df)
        logger.info(f"future_targets_created valid={valid_targets}/{total_records} "
                   f"prediction_horizon={prediction_horizon_hours}h")
        
        return df
    
    def _prepare_base_data(self, df, inference_mode: bool = False):
        """Prepare base data with stations merge and basic processing."""
        stations_df = self.load_stations_data()
        if stations_df is None:
            logger.warning("stations_missing_creating_fallback")
            unique_stations = df['station_id'].unique()
            stations_df = pd.DataFrame({
                'station_id': unique_stations,
                'name': [f'Station_{i}' for i in range(len(unique_stations))],
                'lat': 41.9,
                'lon': -87.6,
                'capacity': 20
            })
        
        df = df.merge(stations_df, on='station_id', how='left', suffixes=('', '_drop'))
        df = df.loc[:, ~df.columns.str.endswith('_drop')]
        
        df['recorded_at'] = pd.to_datetime(df['recorded_at'])
        df = df.sort_values(['station_id', 'recorded_at']).reset_index(drop=True)
        
        if inference_mode:
            # For inference, create the same features as training but no future targets
            df['availability_ratio'] = df['num_bikes_available'] / df['capacity'].fillna(20)
            df['availability_target_current'] = df['availability_ratio'].apply(
                lambda x: 0 if x >= 0.6 else (1 if x >= 0.3 else 2)
            )
            df['availability_target'] = 0  # Dummy value for pipeline compatibility
        else:
            df = self.create_availability_target(df)
            
            # Remove records without valid future targets
            before_filter = len(df)
            df = df.dropna(subset=['availability_target'])
            df['availability_target'] = df['availability_target'].astype(int)
            after_filter = len(df)
            
            logger.info(f"filtered_for_valid_futures removed={before_filter-after_filter} "
                       f"remaining={after_filter}")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessing pipeline and transform data."""
        df = self._prepare_base_data(df)
        
        df = FeatureEngineering.create_all_features(df)
        
        df = self._clean_data(df)
        
        exclude_cols = ['availability_target', 'station_id', 'dow_hour_interaction', 'recorded_at']
        df = self.scaler.fit_transform(df, exclude_cols)
        
        feature_cols = [col for col in df.columns 
                       if col not in ['availability_target', 'station_id'] 
                       and not df[col].dtype.kind in ['M', 'O']]
        X = df[feature_cols]
        y = df['availability_target']
        
        X_selected = self.selector.fit_transform(X, y)
        
        df = pd.concat([
            df[['station_id', 'availability_target']],
            X_selected
        ], axis=1)
        
        self.feature_columns = [col for col in df.columns if col not in ['availability_target', 'station_id']]
        self.is_fitted = True
        
        logger.info(f"preprocessing_fit_transform_complete shape={df.shape} features={len(self.feature_columns)}")
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Single comprehensive data cleaning pass."""
        df = df.copy()
        
        for col in df.columns:
            if col in ['availability_target', 'station_id', 'recorded_at']:
                continue
                
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            elif df[col].dtype == 'object':
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'unknown'
                df[col] = df[col].fillna(fill_val)
        
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['availability_target', 'station_id']:
                df[col] = pd.Categorical(df[col]).codes
        
        return df
    
    def transform(self, df: pd.DataFrame, inference_mode: bool = True) -> pd.DataFrame:
        """Transform data using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        df = self._prepare_base_data(df, inference_mode=inference_mode)
        
        df = FeatureEngineering.create_all_features(df)
        
        df = self._clean_data(df)
        
        exclude_cols = ['availability_target', 'station_id', 'dow_hour_interaction', 'recorded_at']
        df = self.scaler.transform(df, exclude_cols)
        
        feature_cols = [col for col in df.columns 
                       if col not in ['availability_target', 'station_id']
                       and not df[col].dtype.kind in ['M', 'O']]  # Exclude datetime and object types
        X = df[feature_cols]
        X_selected = self.selector.transform(X)
        
        df = pd.concat([
            df[['station_id'] + (['availability_target'] if 'availability_target' in df.columns else [])],
            X_selected
        ], axis=1)
        
        logger.info(f"preprocessing_transform_complete shape={df.shape}")
        return df
    
    def process_training_data(self, days_back: int = 30):
        """Load and process historical data for training."""
        query = """
        SELECT 
            sa.station_id,
            sa.num_bikes_available,
            sa.num_docks_available,
            sa.is_installed,
            sa.is_renting,
            sa.is_returning,
            sa.last_reported,
            sa.recorded_at,
            s.name,
            s.lat,
            s.lon,
            s.capacity
        FROM station_availability sa
        JOIN stations s ON sa.station_id = s.station_id
        WHERE sa.recorded_at >= NOW() - INTERVAL '%s days'
        ORDER BY sa.recorded_at
        """
        
        try:
            logger.info(f"loading_training_data days_back={days_back}")
            df = self.db_client.get_training_data(days_back=days_back)

            logger.info(f"loaded_training_data rows={len(df)} stations={df['station_id'].nunique()}")

            df = self.fit_transform(df)

            drop_cols = ['id', 'recorded_at', 'created_at', 'updated_at', 'last_reported', 'name']
            df = df.drop(columns=[col for col in drop_cols if col in df.columns])
            df = df.dropna(subset=['availability_target'])

            logger.info(f"processed_training_data shape={df.shape}")
            return df

        except Exception as e:
            logger.error(f"training_data_error={e}")
            raise
    
    def process_inference_data(self, recent_dataframes=None):
        """Process data for inference using fitted pipeline."""
        if recent_dataframes is None:
            logger.info("loading_recent_data_from_db")
            db_df = self.db_client.get_recent_availability_data(hours_back=2)
            if not db_df.empty:
                recent_dataframes = [db_df]
            else:
                logger.error("no_recent_data_available")
                return None
        
        df = pd.concat(recent_dataframes, ignore_index=True)
        df = self.transform(df)
        
        drop_cols = ['id', 'recorded_at', 'created_at', 'updated_at', 'last_reported']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        df = df.groupby('station_id').last().reset_index()
        
        logger.info(f"preprocessed_inference shape={df.shape} stations={len(df)}")
        return df
    
    def get_feature_analysis(self, df: pd.DataFrame) -> dict:
        """Get feature analysis on clean data without NaN targets."""
        df = df.dropna(subset=['availability_target'])
        return self.analyzer.analyze_features(df)

