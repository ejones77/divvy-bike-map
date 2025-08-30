import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from divvy_ml.pipelines.feature_engineering import FeatureEngineering
from typing import List

@pytest.fixture
def sample_data():
    """Sample bike availability data for testing."""
    base_time = datetime.now() - timedelta(hours=2)
    data = []
    for i, station_id in enumerate(['123', '456']):
        for j in range(16):  # 4 hours of data at 15min intervals
            data.append({
                'station_id': station_id,
                'num_bikes_available': 5 + i + (j % 4),
                'num_docks_available': 15 - i - (j % 4),
                'is_installed': 1,
                'is_renting': 1,
                'is_returning': 1,
                'recorded_at': base_time + timedelta(minutes=15*j),
                'lat': 41.90 - i*0.05,
                'lon': -87.60 - i*0.05,
                'capacity': 20 + i*5,
                'availability_ratio': (5 + i + (j % 4)) / (20 + i*5)
            })
    return pd.DataFrame(data)

class TestFeatureEngineering:
    
    def test_create_all_features(self, sample_data):
        """Test comprehensive feature engineering."""
        result = FeatureEngineering.create_all_features(sample_data)
        
        # Check basic time features
        assert 'hour_of_day' in result.columns
        assert 'day_of_week' in result.columns
        assert 'is_weekend' in result.columns
        assert 'month' in result.columns
        
        # Check peak indicators
        assert 'is_morning_peak' in result.columns
        assert 'is_evening_peak' in result.columns
        assert 'is_peak_hours' in result.columns
        
        # Check cyclical features
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert 'dow_sin' in result.columns
        assert 'dow_cos' in result.columns
        
        # Check lag features
        assert 'num_bikes_1h_ago' in result.columns
        assert 'num_bikes_3h_ago' in result.columns
        assert 'bikes_change_1h' in result.columns
        assert 'bikes_pct_change_1h' in result.columns
        
        # Check rolling features
        assert 'avg_bikes_2h' in result.columns
        assert 'std_bikes_2h' in result.columns
        assert 'min_bikes_6h' in result.columns
        assert 'max_bikes_12h' in result.columns
        
        # Check capacity features
        assert 'capacity_utilization' in result.columns
        assert 'capacity_pressure' in result.columns
        assert 'bikes_vs_docks_ratio' in result.columns
        assert 'station_reliability' in result.columns
        
        # Check polynomial features
        assert 'availability_ratio_squared' in result.columns
        assert 'availability_ratio_log' in result.columns
        
        # Check interaction features
        assert 'hour_dow_interaction' in result.columns
        assert 'hour_weekend_interaction' in result.columns
        assert 'lat_lon_interaction' in result.columns
        assert 'dow_hour_interaction' in result.columns
        
        # Should have many more features than input
        assert len(result.columns) > len(sample_data.columns) + 30
        
        # Should have same number of rows
        assert len(result) == len(sample_data)
    
    def test_cyclical_features_bounds(self, sample_data):
        """Test that cyclical features are properly bounded."""
        result = FeatureEngineering.create_all_features(sample_data)
        
        # Sin/cos should be between -1 and 1
        assert result['hour_sin'].min() >= -1
        assert result['hour_sin'].max() <= 1
        assert result['hour_cos'].min() >= -1
        assert result['hour_cos'].max() <= 1
        
        assert result['dow_sin'].min() >= -1
        assert result['dow_sin'].max() <= 1
        assert result['dow_cos'].min() >= -1
        assert result['dow_cos'].max() <= 1
    
    def test_lag_features_logic(self, sample_data):
        """Test that lag features work correctly."""
        result = FeatureEngineering.create_all_features(sample_data)
        
        # Check that lag features are filled appropriately
        assert not result['num_bikes_1h_ago'].isna().all()
        assert not result['num_bikes_3h_ago'].isna().all()
        
        # Change features should be calculated correctly
        expected_change = result['num_bikes_available'] - result['num_bikes_1h_ago']
        pd.testing.assert_series_equal(result['bikes_change_1h'], expected_change, check_names=False)
    
    def test_no_missing_critical_columns(self, sample_data):
        """Test that all expected features are created."""
        result = FeatureEngineering.create_all_features(sample_data)
        
        expected_features = [
            'hour_of_day', 'day_of_week', 'is_weekend',
            'num_bikes_1h_ago', 'bikes_change_1h',
            'avg_bikes_2h', 'capacity_utilization',
            'availability_ratio_squared', 'hour_dow_interaction',
            'dow_hour_interaction'
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"