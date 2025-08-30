import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol
from divvy_ml.pipelines.preprocessing import DataPreprocessor
from divvy_ml.utils.model_loader import get_model_path
import logging

logger = logging.getLogger(__name__)


class XGBModel:
    """XGBoost model wrapper with preprocessing pipeline."""
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            self.model_path = get_model_path()
        else:
            self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.preprocessor = None
        self._load_model()
    
    def _load_model(self):
        """Load trained XGBoost model and preprocessing pipeline."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = joblib.load(f"{self.model_path}/model.joblib")
            self.preprocessor = joblib.load(f"{self.model_path}/preprocessor.joblib")
            self.feature_columns = joblib.load(f"{self.model_path}/feature_columns.joblib")
            self.label_encoder = joblib.load(f"{self.model_path}/label_encoders.joblib")
            
            with open(f"{self.model_path}/metadata.json", 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"xgb_model_loaded features={len(self.feature_columns)} "
                       f"trained_at={metadata.get('trained_at', 'unknown')}")
                       
        except Exception as e:
            logger.error(f"xgb_model_load_error={e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for inference using trained preprocessing."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded")
        
        processed_df = self.preprocessor.transform(df, inference_mode=True)
        
        # Ensure all training-time feature columns are present at inference
        # Add any missing columns with a safe default (0)
        if self.feature_columns is None:
            raise ValueError("Feature columns not loaded")
        missing_cols = [col for col in self.feature_columns if col not in processed_df.columns]
        for col in missing_cols:
            processed_df[col] = 0
        
        features = processed_df[self.feature_columns].copy()
        
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            try:
                features[col] = self.label_encoder.transform(features[col].astype(str))
            except ValueError:
                logger.warning(f"unseen_categories_in_{col}_using_fallback")
                features[col] = 0
        
        return features.values
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if isinstance(df, np.ndarray):
            return self.model.predict(df)
        else:
            X = self._prepare_features(df)
            return self.model.predict(X)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if isinstance(df, np.ndarray):
            return self.model.predict_proba(df)
        else:
            X = self._prepare_features(df)
            return self.model.predict_proba(X)

class DivvyPredictor:
    """Unified prediction pipeline."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = XGBModel(model_path)
        
        self.data_loader = DataPreprocessor(
            n_features=30,
            use_scaling=False
        )
    
    def _load_raw_data(self):
        """Load raw data without preprocessing."""
        try:
            db_df = self.data_loader.db_client.get_recent_availability_data(hours_back=2)
            if not db_df.empty:
                return db_df
            else:
                raise ValueError("No recent data available")
        except Exception as e:
            logger.error(f"data_loading_failed error={e}")
            raise
    
    def _generate_predictions(self, raw_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate predictions using the loaded model."""
        try:
            logger.info(f"predict_input shape={raw_data.shape}")
            
            if raw_data.empty:
                raise ValueError("No data provided for prediction")
            
            processed_data = self.model.preprocessor.transform(raw_data, inference_mode=True)
            if processed_data is None or processed_data.empty:
                raise ValueError("Preprocessing failed or returned empty data")
            
            feature_data = processed_data.drop(columns=['station_id'] if 'station_id' in processed_data.columns else [])
            
            predictions = self.model.predict(feature_data)
            probabilities = self.model.predict_proba(feature_data)
            
            result_df = pd.DataFrame({
                'station_id': processed_data['station_id'],
                'predicted_availability_class': predictions.astype(int),
                'prediction_time': datetime.now(timezone.utc) + pd.Timedelta(hours=6),
                'horizon_hours': 6
            })
            
            result_df['confidence_green'] = probabilities[:, 0]
            result_df['confidence_yellow'] = probabilities[:, 1] 
            result_df['confidence_red'] = probabilities[:, 2]
            
            class_labels = {0: 'green', 1: 'yellow', 2: 'red'}
            result_df['availability_prediction'] = result_df['predicted_availability_class'].map(class_labels)
            
            class_counts = result_df['availability_prediction'].value_counts().to_dict()
            logger.info(f"prediction_distribution={class_counts}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"prediction_error={e}")
            raise
    
    def run_inference(self) -> pd.DataFrame:
        """Run complete inference pipeline."""
        try:
            raw_data = self._load_raw_data()
            predictions = self._generate_predictions(raw_data)
            logger.info(f"inference_complete predictions={len(predictions)}")
            return predictions
            
        except Exception as e:
            logger.error(f"inference_failed error={e}")
            raise

def main():
    predictor = DivvyPredictor()
    predictions = predictor.run_inference()
    logger.info(f"main_complete predictions={len(predictions)}")

if __name__ == "__main__":
    main()
