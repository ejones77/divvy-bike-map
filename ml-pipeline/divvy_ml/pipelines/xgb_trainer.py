import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import json
from typing import Tuple, Dict, Optional
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from divvy_ml.pipelines.preprocessing import DataPreprocessor
from divvy_ml.utils.model_loader import find_local_model_directory
import logging

logger = logging.getLogger(__name__)

class XGBTrainer:
    def __init__(self, model_path: Optional[str] = None, n_features: int = 30):
        if model_path is None:
            timestamp = datetime.now().strftime("%m-%d-%y")
            model_path = f"xgb_model_{timestamp}"
        self.model_path = model_path
        self.preprocessor = DataPreprocessor(n_features=n_features, use_scaling=True)
        
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.best_params = None
        self.is_trained = False
        
    def _prepare_features(self, df: pd.DataFrame, fit_transforms: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix - assumes data is already clean from preprocessing."""
        target = df['availability_target'].astype(int)
        
        feature_cols = [col for col in df.columns if col not in ['availability_target', 'station_id']]
        features = df[feature_cols]
        
        if fit_transforms:
            self.feature_columns = feature_cols
        
        logger.info(f"features_prepared shape={features.shape} target_classes={sorted(target.unique())}")
        return features.values, target.values
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Simple grid search with temporal CV."""
        logger.info("Starting simple hyperparameter tuning with temporal validation")
        
        param_grid = [
            {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 200},
            {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 250},
            {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200},
            {'max_depth': 6, 'learning_rate': 0.15, 'n_estimators': 250},
            {'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 300},
        ]
        
        best_score = -1
        best_params = None
        
        # Use temporal validation (no shuffling, respect time order)
        n_splits = 3
        fold_size = len(X_train) // (n_splits + 1)
        
        for params in param_grid:
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                **params
            )
            
            scores = []
            for i in range(n_splits):
                train_end = (i + 1) * fold_size
                val_start = train_end
                val_end = val_start + fold_size
                
                X_fold_train = X_train[:train_end]
                y_fold_train = y_train[:train_end]
                X_fold_val = X_train[val_start:val_end]
                y_fold_val = y_train[val_start:val_end]
                
                if len(X_fold_val) == 0:
                    continue
                    
                model.fit(X_fold_train, y_fold_train)
                score = model.score(X_fold_val, y_fold_val)
                scores.append(score)
            
            mean_score = np.mean(scores) if scores else 0
            logger.info(f"params={params} temporal_cv_score={mean_score:.3f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        logger.info(f"best_temporal_cv_score={best_score:.3f} best_params={best_params}")
        
        return {
            **best_params,
            'objective': 'multi:softprob',
            'num_class': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }

    def train(self, days_back: int = 30, test_size: float = 0.2, 
              tune_hyperparams: bool = True, n_iter: int = 25) -> dict:
        """Train XGBoost model efficiently."""
        try:
            logger.info(f"training_started days_back={days_back} tune_hyperparams={tune_hyperparams}")
            

            processed_data = self.preprocessor.process_training_data(days_back)
            X, y = self._prepare_features(processed_data, fit_transforms=True)
            
            feature_analysis = self.preprocessor.get_feature_analysis(processed_data)
            
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"temporal_split train_shape={X_train.shape} test_shape={X_test.shape}")
            logger.info(f"test_period_is_most_recent_{test_size*100:.0f}%_of_data")
            
            if tune_hyperparams:
                best_params = self._tune_hyperparameters(X_train, y_train)
                self.best_params = best_params
            else:
                best_params = {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 200,
                    'random_state': 42,
                    'n_jobs': -1
                }
            
            best_params['early_stopping_rounds'] = 20
            
            self.model = xgb.XGBClassifier(**best_params)
            
            logger.info(f"fitting_model train_shape={X_train.shape} test_shape={X_test.shape}")
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            y_pred = self.model.predict(X_test)
            
            n_splits = 5
            fold_size = len(X_train) // (n_splits + 1)
            cv_scores = []
            
            for i in range(n_splits):
                train_end = (i + 1) * fold_size
                val_start = train_end
                val_end = val_start + fold_size
                
                if val_end > len(X_train):
                    break
                    
                X_fold_train = X_train[:train_end]
                y_fold_train = y_train[:train_end]
                X_fold_val = X_train[val_start:val_end]
                y_fold_val = y_train[val_start:val_end]
                
                fold_model = xgb.XGBClassifier(**best_params)
                fold_model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)])
                score = fold_model.score(X_fold_val, y_fold_val)
                cv_scores.append(score)
            
            cv_scores = np.array(cv_scores)
            
            model_feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
            
            sorted_importance = sorted(model_feature_importance.items(), key=lambda x: x[1], reverse=True)
            logger.info("model_feature_importance (top 10):")
            for i, (feature, importance) in enumerate(sorted_importance[:10]):
                logger.info(f"  {i+1:2d}. {feature:<25} importance={importance:.4f}")
            
            metrics = {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'test_accuracy': accuracy_score(y_test, y_pred),
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'model_feature_importance': sorted(model_feature_importance.items(), 
                                                 key=lambda x: x[1], reverse=True)[:10],
                'preprocessing_analysis': feature_analysis,
                'best_params': best_params if tune_hyperparams else None
            }
            
            self.is_trained = True
            self.save_model()
            
            logger.info(f"training_completed test_accuracy={metrics['test_accuracy']:.3f} "
                       f"cv_accuracy={metrics['cv_accuracy_mean']:.3f}±{metrics['cv_accuracy_std']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"training_failed error={e}")
            raise
    
    def save_model(self):
        """Save trained model and preprocessing pipeline."""
        os.makedirs(self.model_path, exist_ok=True)
        
        joblib.dump(self.model, f"{self.model_path}/model.joblib")
        joblib.dump(self.label_encoders, f"{self.model_path}/label_encoders.joblib")
        joblib.dump(self.feature_columns, f"{self.model_path}/feature_columns.joblib")
        
        if hasattr(self.preprocessor, '__dict__'):
            preprocessor_copy = self.preprocessor.__class__.__new__(self.preprocessor.__class__)
            preprocessor_copy.__dict__ = {k: v for k, v in self.preprocessor.__dict__.items() 
                                         if k not in ['s3_client', 'db_client']}
        else:
            preprocessor_copy = self.preprocessor
        
        joblib.dump(preprocessor_copy, f"{self.model_path}/preprocessor.joblib")
        
        metadata = {
            'model_type': 'xgboost',
            'version': '2.0',
            'trained_at': datetime.now().isoformat(),
            'feature_count': len(self.feature_columns),
            'target_classes': [0, 1, 2],
            'best_params': self.best_params,
            'n_selected_features': self.preprocessor.n_features
        }
        
        with open(f"{self.model_path}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"model_saved path={self.model_path}")

def main():
    trainer = XGBTrainer(n_features=30)
    metrics = trainer.train(days_back=30, tune_hyperparams=True, n_iter=25)
    
    print(f"Training completed:")
    print(f"  Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"  CV accuracy: {metrics['cv_accuracy_mean']:.3f}±{metrics['cv_accuracy_std']:.3f}")
    print(f"  Best params: {metrics['best_params']}")

if __name__ == "__main__":
    main()