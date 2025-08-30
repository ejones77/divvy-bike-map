import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import tempfile
import os
from divvy_ml.config import ml_config

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
        for j in range(8):
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
def mock_s3_client():
    mock_client = Mock()
    mock_client.list_objects_v2.return_value = {
        'Contents': [
            {'Key': f'{ml_config.incremental_path}20240101_1200.csv'},
            {'Key': f'{ml_config.incremental_path}20240101_1215.csv'}
        ]
    }
    mock_client.get_object.return_value = {'Body': Mock()}
    mock_client.upload_file.return_value = None
    mock_client.head_object.return_value = {}
    return mock_client

@pytest.fixture
def mock_s3_service(sample_stations_data, sample_availability_data, mock_s3_client):
    with patch('divvy_ml.utils.s3_client.boto3.client') as mock_boto:
        mock_boto.return_value = mock_s3_client
        
        def mock_download_csv(bucket, key):
            if 'stations.csv' in key:
                return sample_stations_data
            return sample_availability_data
        
        with patch('divvy_ml.utils.s3_client.S3Client.download_csv_to_dataframe', side_effect=mock_download_csv):
            yield

@pytest.fixture
def temp_model_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'canvas_trained_model')
        os.makedirs(model_path)
        yield model_path

@pytest.fixture
def mock_db_client(sample_stations_data, sample_availability_data):
    mock_client = Mock()
    mock_client.get_stations_metadata.return_value = sample_stations_data
    mock_client.get_recent_availability_data.return_value = sample_availability_data
    return mock_client

@pytest.fixture
def mock_autogluon():
    mock_predictor = Mock()
    mock_predictor.problem_type = 'multiclass'
    mock_predictor.predict.return_value = np.array([0, 1, 2])
    mock_predictor._learner.predict.return_value = np.array([0, 1, 2])
    
    with patch('autogluon.tabular.TabularPredictor') as mock_class:
        mock_class.load.return_value = mock_predictor
        yield mock_predictor

@pytest.fixture
def mock_file_operations():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink, \
             patch('tempfile.NamedTemporaryFile') as mock_temp:
            
            mock_exists.return_value = True
            temp_file = os.path.join(temp_dir, 'test.csv')
            mock_temp.return_value.__enter__.return_value.name = temp_file
            yield {
                'exists': mock_exists,
                'unlink': mock_unlink,
                'temp': mock_temp
            }