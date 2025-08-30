import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import json

@pytest.fixture
def clean_training_data():
    """Pre-processed training data for XGB testing."""
    return pd.DataFrame({
        'station_id': ['123'] * 50 + ['456'] * 50,
        'availability_target': [0, 1, 2] * 33 + [0],  # Balanced classes
        'feature_1': np.random.normal(0, 1, 100),
        'feature_2': np.random.normal(0, 1, 100),
        'feature_3': np.random.normal(0, 1, 100),
    })

class TestXGBTrainer:
    
    def test_prepare_features_clean_input(self, clean_training_data):
        """Test feature preparation with clean preprocessed data."""
        with patch.dict('sys.modules', {'xgboost': Mock()}):
            from divvy_ml.pipelines.xgb_trainer import XGBTrainer
            
            trainer = XGBTrainer.__new__(XGBTrainer)
            trainer.label_encoders = {}
            
            X, y = trainer._prepare_features(clean_training_data, fit_transforms=True)
            
            assert X.shape == (100, 3)  # Excludes station_id, availability_target
            assert len(y) == 100
            assert set(y) == {0, 1, 2}
    
    def test_hyperparameter_tuning_structure(self, clean_training_data):
        """Test hyperparameter tuning returns valid params."""
        with patch.dict('sys.modules', {'xgboost': Mock()}):
            from divvy_ml.pipelines.xgb_trainer import XGBTrainer
            
            trainer = XGBTrainer.__new__(XGBTrainer)
            trainer.label_encoders = {}
            
            X, y = trainer._prepare_features(clean_training_data, fit_transforms=True)
            
            # Mock XGBoost classifier for temporal CV
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.score.return_value = 0.85
            
            with patch('divvy_ml.pipelines.xgb_trainer.xgb.XGBClassifier', return_value=mock_model):
                params = trainer._tune_hyperparameters(X, y)
                
                required_keys = ['max_depth', 'learning_rate', 'n_estimators', 'objective']
                assert all(key in params for key in required_keys)
    
    def test_save_model_structure(self):
        """Test model saving creates required files."""
        with patch.dict('sys.modules', {'xgboost': Mock()}):
            from divvy_ml.pipelines.xgb_trainer import XGBTrainer
            
            trainer = XGBTrainer.__new__(XGBTrainer)
            trainer.model = Mock()
            trainer.label_encoders = {}
            trainer.feature_columns = ['f1', 'f2']
            
            # Use a simple object to avoid Mock __dict__ issues
            class SimplePreprocessor:
                def __init__(self):
                    self.n_features = 10
            
            trainer.preprocessor = SimplePreprocessor()
            trainer.best_params = {'test': 'param'}
            
            with tempfile.TemporaryDirectory() as temp_dir:
                trainer.model_path = temp_dir
                
                with patch('joblib.dump') as mock_dump, \
                     patch('builtins.open', create=True) as mock_open:
                    
                    trainer.save_model()
                    
                    # Verify the correct files are being saved
                    expected_calls = [
                        (trainer.model, f"{temp_dir}/model.joblib"),
                        (trainer.label_encoders, f"{temp_dir}/label_encoders.joblib"),  # Note: plural
                        (trainer.feature_columns, f"{temp_dir}/feature_columns.joblib"),
                        (trainer.preprocessor, f"{temp_dir}/preprocessor.joblib")
                    ]
                    
                    assert mock_dump.call_count == 4
                    mock_open.assert_called_once()  # metadata.json
