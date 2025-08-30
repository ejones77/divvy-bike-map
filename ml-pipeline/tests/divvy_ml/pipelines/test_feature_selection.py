import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from divvy_ml.pipelines.feature_selection import FeatureSelector, FeatureScaler, FeatureAnalyzer
from typing import List
import logging

logger = logging.getLogger(__name__)

@pytest.fixture
def clean_feature_data():
    """Pre-cleaned feature data for selection testing."""
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=10, 
        n_redundant=5, n_classes=3, random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(X, columns=feature_names)
    target = pd.Series(y, name='target')
    return df, target

class TestFeatureSelector:
    
    def test_fit_transform_basic(self, clean_feature_data):
        X, y = clean_feature_data
        selector = FeatureSelector(n_features=10)
        
        result = selector.fit_transform(X, y)
        
        assert result.shape == (100, 10)
        assert len(selector.selected_features) == 10
        assert selector.selector is not None
    
    def test_transform_requires_fit(self, clean_feature_data):
        X, y = clean_feature_data
        selector = FeatureSelector(n_features=10)
        
        with pytest.raises(ValueError, match="not fitted"):
            selector.transform(X)
    
    def test_transform_after_fit(self, clean_feature_data):
        X, y = clean_feature_data
        selector = FeatureSelector(n_features=5)
        
        selector.fit_transform(X, y)
        result = selector.transform(X)
        
        assert result.shape == (100, 5)
        assert list(result.columns) == selector.selected_features

class TestFeatureScaler:
    def test_robust_scaling(self, clean_feature_data):
        X, _ = clean_feature_data
        scaler = FeatureScaler()
        
        result = scaler.fit_transform(X)
        
        assert result.shape == X.shape
        assert not result.equals(X)  # Should be different after scaling
    
    def test_exclude_columns(self, clean_feature_data):
        X, _ = clean_feature_data
        X['station_id'] = ['123'] * 100
        scaler = FeatureScaler()
        
        result = scaler.fit_transform(X, exclude_cols=['station_id'])
        
        assert result['station_id'].equals(X['station_id'])  # Unchanged

class TestFeatureAnalyzer:
    
    def test_analyze_features(self, clean_feature_data):
        X, y = clean_feature_data
        df = X.copy()
        df['availability_target'] = y
        
        analysis = FeatureAnalyzer.analyze_features(df)
        
        assert 'feature_importance' in analysis
        assert 'total_features' in analysis
        assert analysis['total_features'] == 20