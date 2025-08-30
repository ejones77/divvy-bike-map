import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import os
from divvy_ml.pipelines.preprocessing import DataPreprocessor

@pytest.fixture(autouse=True)
def mock_db_url():
    """Automatically mock DB_URL for all tests"""
    with patch.dict(os.environ, {'DB_URL': 'postgresql://test:test@test:5432/test'}):
        yield

@pytest.fixture
def sample_stations_data():
    return pd.DataFrame({
        'station_id': ['123', '456', '789'],
        'name': ['Station A', 'Station B', 'Station C'],
        'lat': [41.90, 41.85, 41.88],
        'lon': [-87.60, -87.65, -87.62],
        'capacity': [20, 15, 25]
    })

@pytest.fixture
def sample_availability_data():
    base_time = datetime.now() - timedelta(hours=2)
    data = []
    for i, station_id in enumerate(['123', '456', '789']):
        for j in range(16):  # More data points for rolling features
            data.append({
                'station_id': station_id,
                'num_bikes_available': 5 + i + (j % 3),
                'num_docks_available': 15 - i - (j % 3),
                'is_installed': 1,
                'is_renting': 1,
                'is_returning': 1,
                'last_reported': int((base_time + timedelta(minutes=15*j)).timestamp()),
                'recorded_at': base_time + timedelta(minutes=15*j),
                'name': f'Station {chr(65+i)}',
                'lat': 41.90 - i*0.05,
                'lon': -87.60 - i*0.05,
                'capacity': 20 - i*5
            })
    return pd.DataFrame(data)

@pytest.fixture
def mock_feature_data():
    """Create valid feature data for testing sklearn components."""
    return pd.DataFrame({
        'station_id': ['123', '456', '789'] * 8,  # 24 samples total
        'feature1': np.random.rand(24),
        'feature2': np.random.rand(24),
        'feature3': np.random.rand(24),
        'feature4': np.random.rand(24),
        'availability_target': [0, 1, 2] * 8
    })

@pytest.fixture
def base_availability_data():
    """Minimal availability data for testing with enough time span for future targets."""
    base_time = datetime.now() - timedelta(hours=8)
    return pd.DataFrame([
        {
            'station_id': '123',
            'num_bikes_available': 5 + (i % 3),
            'num_docks_available': 15 - (i % 3),
            'is_installed': 1,
            'is_renting': 1,
            'is_returning': 1,
            'recorded_at': base_time + timedelta(hours=i),
            'capacity': 20
        }
        for i in range(10)
    ])

@pytest.fixture
def mock_pipeline_components():
    """Set up properly mocked pipeline components with required methods."""
    with patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
         patch('divvy_ml.pipelines.preprocessing.FeatureSelector') as mock_sel, \
         patch('divvy_ml.pipelines.preprocessing.FeatureScaler') as mock_sc, \
         patch('divvy_ml.pipelines.preprocessing.FeatureAnalyzer') as mock_an:
        
        # Create mock instances
        mock_fe_instance = Mock()
        mock_sel_instance = Mock()
        mock_sc_instance = Mock()
        mock_an_instance = Mock()
        
        mock_fe.create_all_features = Mock()
        mock_sel_instance.fit_transform = Mock()
        mock_sel_instance.transform = Mock()
        mock_sc_instance.fit_transform = Mock()
        mock_sc_instance.transform = Mock()
        mock_an_instance.analyze_features = Mock()
        

        mock_sel.return_value = mock_sel_instance
        mock_sc.return_value = mock_sc_instance
        mock_an.return_value = mock_an_instance
        
        yield {
            'feature_engineering': mock_fe,
            'selector': mock_sel_instance,
            'scaler': mock_sc_instance,
            'analyzer': mock_an_instance
        }

@pytest.fixture
def mock_preprocessor_dependencies():
    """Mock all external dependencies."""
    with patch('divvy_ml.pipelines.preprocessing.DatabaseClient') as mock_db, \
         patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
         patch('divvy_ml.pipelines.preprocessing.FeatureSelector') as mock_sel, \
         patch('divvy_ml.pipelines.preprocessing.FeatureScaler') as mock_sc, \
         patch('divvy_ml.pipelines.preprocessing.FeatureAnalyzer') as mock_an:
        
        # Create mock instances with required methods
        mock_fe.create_all_features = Mock()
        
        mock_sel_instance = Mock()
        mock_sel_instance.fit_transform = Mock()
        mock_sel_instance.transform = Mock()
        
        mock_sc_instance = Mock()
        mock_sc_instance.fit_transform = Mock()
        mock_sc_instance.transform = Mock()
        
        mock_an_instance = Mock()
        mock_an_instance.analyze_features = Mock()
        
        # Set up the mock returns
        mock_sel.return_value = mock_sel_instance
        mock_sc.return_value = mock_sc_instance
        mock_an.return_value = mock_an_instance
        
        mock_db.return_value = Mock()
        
        yield

class TestDataPreprocessor:
    
    def test_init(self, mock_preprocessor_dependencies):
        with patch('divvy_ml.pipelines.preprocessing.DatabaseClient') as mock_db:
            mock_db.return_value = Mock()
            with patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
                 patch('divvy_ml.pipelines.preprocessing.FeatureSelector') as mock_sel, \
                 patch('divvy_ml.pipelines.preprocessing.FeatureScaler') as mock_sc, \
                 patch('divvy_ml.pipelines.preprocessing.FeatureAnalyzer') as mock_an:
                
                mock_fe.create_all_features = Mock()
                mock_sel.return_value = Mock()
                mock_sc.return_value = Mock()
                mock_an.return_value = Mock()
                
                processor = DataPreprocessor(n_features=20, use_scaling=False)
                
                assert processor.n_features == 20
                assert not processor.use_scaling
                assert not processor.is_fitted
    
    def test_create_availability_target(self, mock_preprocessor_dependencies):
        """Test future availability target creation (6 hours ahead)."""
        with patch('divvy_ml.pipelines.preprocessing.DatabaseClient') as mock_db, \
             patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
             patch('divvy_ml.pipelines.preprocessing.FeatureSelector') as mock_sel, \
             patch('divvy_ml.pipelines.preprocessing.FeatureScaler') as mock_sc, \
             patch('divvy_ml.pipelines.preprocessing.FeatureAnalyzer') as mock_an:
            
            mock_db.return_value = Mock()
            mock_fe.return_value = Mock()
            mock_sel.return_value = Mock()
            mock_sc.return_value = Mock()
            mock_an.return_value = Mock()
            
            processor = DataPreprocessor()
            
            # Create test data with multiple time points for the same station
            base_time = datetime.now()
            df = pd.DataFrame({
                'station_id': ['123'] * 8 + ['456'] * 8,
                'num_bikes_available': [0, 5, 12, 18, 8, 15, 3, 10, 2, 7, 14, 16, 9, 12, 4, 11],
                'capacity': [20] * 16,
                'recorded_at': [base_time + timedelta(hours=i) for i in range(8)] * 2
            })
            
            result = processor.create_availability_target(df)
            
            # Basic columns should exist
            assert 'availability_ratio' in result.columns
            assert 'availability_target' in result.columns
            assert 'availability_target_current' in result.columns
    
    def test_fit_transform_pipeline_order(self, mock_preprocessor_dependencies, base_availability_data):
        """Test that pipeline components are called in correct order."""
        with patch('divvy_ml.pipelines.preprocessing.DatabaseClient') as mock_db, \
             patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
             patch('divvy_ml.pipelines.preprocessing.FeatureSelector') as mock_sel, \
             patch('divvy_ml.pipelines.preprocessing.FeatureScaler') as mock_sc, \
             patch('divvy_ml.pipelines.preprocessing.FeatureAnalyzer') as mock_an:
            
            mock_db.return_value = Mock()
            mock_fe.return_value = Mock()
            mock_sel.return_value = Mock()
            mock_sc.return_value = Mock()
            mock_an.return_value = Mock()
            
            processor = DataPreprocessor()
            
            # First add availability_target to base data
            base_availability_data['availability_target'] = 0  # Add dummy target

            # Then set up the mock returns with this modified data
            processor.scaler.fit_transform.return_value = base_availability_data.copy()
            processor.selector.fit_transform.return_value = base_availability_data.copy()
            
            with patch.object(processor, 'load_stations_data') as mock_stations, \
                 patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe:
                mock_stations.return_value = pd.DataFrame({
                    'station_id': ['123'],
                    'name': ['Test Station'],
                    'lat': [41.9], 'lon': [-87.6], 'capacity': [20]
                })
                mock_fe.create_all_features.return_value = base_availability_data.copy()
                
                result = processor.fit_transform(base_availability_data)
                
                # Verify pipeline order
                mock_fe.create_all_features.assert_called_once()
                processor.scaler.fit_transform.assert_called_once()
                processor.selector.fit_transform.assert_called_once()
                assert processor.is_fitted
    
    @patch('divvy_ml.pipelines.preprocessing.DatabaseClient')
    def test_load_stations_data_success(self, mock_db_class, sample_stations_data):
        """Test successful stations data loading from database."""
        # Mock successful DB call
        mock_db_instance = Mock()
        mock_db_instance.get_stations_metadata.return_value = sample_stations_data
        mock_db_class.return_value = mock_db_instance
        
        with patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
             patch('divvy_ml.pipelines.preprocessing.FeatureSelector') as mock_sel, \
             patch('divvy_ml.pipelines.preprocessing.FeatureScaler') as mock_sc, \
             patch('divvy_ml.pipelines.preprocessing.FeatureAnalyzer') as mock_an:
            
            mock_fe.return_value = Mock()
            mock_sel.return_value = Mock()
            mock_sc.return_value = Mock()
            mock_an.return_value = Mock()
            
            preprocessor = DataPreprocessor()
            result = preprocessor.load_stations_data()
            
            pd.testing.assert_frame_equal(result, sample_stations_data)
            mock_db_instance.get_stations_metadata.assert_called_once()
    
    @patch('divvy_ml.pipelines.preprocessing.DatabaseClient')
    def test_load_stations_data_failure(self, mock_db_class):
        """Test stations data loading with DB failure."""
        # Mock DB failure
        mock_db_instance = Mock()
        mock_db_instance.get_stations_metadata.side_effect = Exception("DB connection failed")
        mock_db_class.return_value = mock_db_instance
        
        with patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
             patch('divvy_ml.pipelines.preprocessing.FeatureSelector') as mock_sel, \
             patch('divvy_ml.pipelines.preprocessing.FeatureScaler') as mock_sc, \
             patch('divvy_ml.pipelines.preprocessing.FeatureAnalyzer') as mock_an:
            
            mock_fe.return_value = Mock()
            mock_sel.return_value = Mock()
            mock_sc.return_value = Mock()
            mock_an.return_value = Mock()
            
            preprocessor = DataPreprocessor()
            
            with pytest.raises(RuntimeError, match="Failed to load stations data from database"):
                preprocessor.load_stations_data()
    
    @patch('divvy_ml.pipelines.preprocessing.DatabaseClient')
    def test_load_availability_data_success(self, mock_db_class, sample_availability_data):
        """Test successful availability data loading."""
        # Mock successful DB call
        mock_db_instance = Mock()
        mock_db_instance.get_availability_data.return_value = sample_availability_data
        mock_db_class.return_value = mock_db_instance
        
        with patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
             patch('divvy_ml.pipelines.preprocessing.FeatureSelector') as mock_sel, \
             patch('divvy_ml.pipelines.preprocessing.FeatureScaler') as mock_sc, \
             patch('divvy_ml.pipelines.preprocessing.FeatureAnalyzer') as mock_an:
            
            mock_fe.return_value = Mock()
            mock_sel.return_value = Mock()
            mock_sc.return_value = Mock()
            mock_an.return_value = Mock()
            
            preprocessor = DataPreprocessor()
            result = preprocessor._load_availability_data(hours_back=24)
            
            pd.testing.assert_frame_equal(result, sample_availability_data)
            mock_db_instance.get_availability_data.assert_called_once_with(hours_back=24, inference_mode=True)
    
    @patch('divvy_ml.pipelines.preprocessing.DatabaseClient')
    def test_load_availability_data_failure(self, mock_db_class):
        """Test availability data loading failure."""
        # Mock DB failure
        mock_db_instance = Mock()
        mock_db_instance.get_availability_data.side_effect = Exception("DB connection failed")
        mock_db_class.return_value = mock_db_instance
        
        with patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
             patch('divvy_ml.pipelines.preprocessing.FeatureSelector') as mock_sel, \
             patch('divvy_ml.pipelines.preprocessing.FeatureScaler') as mock_sc, \
             patch('divvy_ml.pipelines.preprocessing.FeatureAnalyzer') as mock_an:
            
            mock_fe.return_value = Mock()
            mock_sel.return_value = Mock()
            mock_sc.return_value = Mock()
            mock_an.return_value = Mock()
            
            preprocessor = DataPreprocessor()
            
            with pytest.raises(Exception):
                preprocessor._load_availability_data()
    
    def test_fit_transform(self, mock_pipeline_components, mock_feature_data):
        """Test fit and transform pipeline."""
        processor = DataPreprocessor()
        
        # Set up mock returns
        mock_pipeline_components['feature_engineering'].create_all_features.return_value = mock_feature_data
        mock_pipeline_components['scaler'].fit_transform.return_value = mock_feature_data
        mock_pipeline_components['selector'].fit_transform.return_value = mock_feature_data
        
        with patch.object(processor, '_prepare_base_data') as mock_prepare:
            mock_prepare.return_value = mock_feature_data
            result = processor.fit_transform(mock_feature_data)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert processor.is_fitted is True

    def test_transform_inference_mode(self, mock_preprocessor_dependencies, sample_availability_data):
        """Test transform with inference_mode=True skips target filtering."""
        with patch('divvy_ml.pipelines.preprocessing.DatabaseClient') as mock_db, \
             patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
             patch('divvy_ml.pipelines.preprocessing.FeatureSelector') as mock_sel, \
             patch('divvy_ml.pipelines.preprocessing.FeatureScaler') as mock_sc, \
             patch('divvy_ml.pipelines.preprocessing.FeatureAnalyzer') as mock_an:
            
            mock_db.return_value = Mock()
            mock_fe.return_value = Mock()
            mock_sel.return_value = Mock()
            mock_sc.return_value = Mock()
            mock_an.return_value = Mock()
            
            processor = DataPreprocessor()
            processor.is_fitted = True
            
            # Mock the pipeline components
            with patch.object(processor, 'load_stations_data') as mock_stations, \
                 patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
                 patch.object(processor.scaler, 'transform') as mock_scaler, \
                 patch.object(processor.selector, 'transform') as mock_selector:
                
                # Mock stations data
                mock_stations.return_value = pd.DataFrame({
                    'station_id': ['123', '456', '789'],
                    'name': ['Station A', 'Station B', 'Station C'],
                    'lat': [41.9, 41.85, 41.88],
                    'lon': [-87.6, -87.65, -87.62],
                    'capacity': [20, 15, 25]
                })
                
                # Create test data
                test_data = sample_availability_data.copy()
                
                # Mock pipeline returns
                mock_fe.create_all_features.return_value = test_data
                mock_scaler.return_value = test_data
                mock_selector.return_value = pd.DataFrame({
                    'station_id': range(len(test_data)),
                    'feature1': range(len(test_data)),
                    'feature2': range(len(test_data)),
                    'availability_target': [0] * len(test_data)  # Add target column
                })
                
                # Test inference mode - should not filter out data
                result = processor.transform(test_data, inference_mode=True)
                
                assert isinstance(result, pd.DataFrame)
                assert len(result) > 0
                assert 'availability_target' in result.columns
    
    def test_transform_training_mode(self, mock_preprocessor_dependencies, base_availability_data):
        """Test transform with inference_mode=False does target filtering."""
        with patch('divvy_ml.pipelines.preprocessing.DatabaseClient') as mock_db, \
             patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
             patch('divvy_ml.pipelines.preprocessing.FeatureSelector') as mock_sel, \
             patch('divvy_ml.pipelines.preprocessing.FeatureScaler') as mock_sc, \
             patch('divvy_ml.pipelines.preprocessing.FeatureAnalyzer') as mock_an:
            
            mock_db.return_value = Mock()
            mock_fe.return_value = Mock()
            mock_sel.return_value = Mock()
            mock_sc.return_value = Mock()
            mock_an.return_value = Mock()
            
            processor = DataPreprocessor()
            processor.is_fitted = True
            
            with patch.object(processor, 'load_stations_data') as mock_stations, \
                 patch.object(processor, 'create_availability_target') as mock_target, \
                 patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe_transform, \
                 patch.object(processor.scaler, 'transform') as mock_scaler, \
                 patch.object(processor.selector, 'transform') as mock_selector:
                
                mock_stations.return_value = pd.DataFrame({
                    'station_id': ['123'],
                    'name': ['Test Station'],
                    'lat': [41.9], 'lon': [-87.6], 'capacity': [20]
                })
                
                input_data = base_availability_data.copy()
                
                # Create data with NaN targets returned by create_availability_target
                test_data = base_availability_data.copy()
                test_data['availability_target'] = [0, 1] + [np.nan] * (len(test_data) - 2)
                mock_target.return_value = test_data
                
                # After _prepare_base_data filtering, only 2 rows remain
                filtered_data = pd.DataFrame({
                    'station_id': ['123', '123'],
                    'availability_target': [0, 1]
                })
                
                # Mock pipeline components to return filtered data
                mock_fe_transform.create_all_features.return_value = filtered_data.copy()
                mock_scaler.return_value = filtered_data.copy()
                
                final_data = pd.DataFrame({
                    'station_id': ['123', '123'],
                    'feature1': [0, 1],
                    'feature2': [0, 1],
                    'availability_target': [0, 1]
                })
                mock_selector.return_value = final_data
                
                result = processor.transform(input_data, inference_mode=False)
                
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 2
                assert len(test_data) == 10
                assert len(result) < len(test_data)
    
    def test_prepare_base_data_inference_mode(self, mock_preprocessor_dependencies):
        """Test _prepare_base_data with inference_mode parameter."""
        with patch('divvy_ml.pipelines.preprocessing.DatabaseClient') as mock_db, \
             patch('divvy_ml.pipelines.preprocessing.FeatureEngineering') as mock_fe, \
             patch('divvy_ml.pipelines.preprocessing.FeatureSelector') as mock_sel, \
             patch('divvy_ml.pipelines.preprocessing.FeatureScaler') as mock_sc, \
             patch('divvy_ml.pipelines.preprocessing.FeatureAnalyzer') as mock_an:
            
            mock_db.return_value = Mock()
            mock_fe.return_value = Mock()
            mock_sel.return_value = Mock()
            mock_sc.return_value = Mock()
            mock_an.return_value = Mock()
            
            processor = DataPreprocessor()
            
            test_data = pd.DataFrame({
                'station_id': ['123', '456'],
                'num_bikes_available': [5, 3],
                'recorded_at': [datetime.now(), datetime.now()]
            })
            
            with patch.object(processor, 'load_stations_data') as mock_stations:
                mock_stations.return_value = pd.DataFrame({
                    'station_id': ['123', '456'],
                    'name': ['Station A', 'Station B'],
                    'lat': [41.9, 41.85],
                    'lon': [-87.6, -87.65],
                    'capacity': [20, 15]
                })
                
                # Test inference mode
                result_inference = processor._prepare_base_data(test_data, inference_mode=True)
                assert 'availability_target' in result_inference.columns
                assert all(result_inference['availability_target'].fillna(-1) >= 0)  # Allow for NaN values