import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from typing import List, Dict
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Focused feature selection using mutual information."""
    
    def __init__(self, n_features: int = 30):
        self.n_features = n_features
        self.selector = None
        self.selected_features = None
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit selector and transform features - assumes clean data."""
        self.selector = SelectKBest(f_classif, k=min(self.n_features, len(X.columns)))
        X_selected = self.selector.fit_transform(X, y)
        
        selected_indices = self.selector.get_support(indices=True)
        self.selected_features = [X.columns[i] for i in selected_indices]
        

        f_scores = self.selector.scores_[selected_indices]
        
        feature_scores = list(zip(self.selected_features, f_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        top_n = 5
        top_items = ", ".join([f"{feat}={score:.3f}" for feat, score in feature_scores[:top_n]])
        logger.info(
            f"feature_selection selected={len(self.selected_features)}/{len(X.columns)} by=ANOVA_F f_top{top_n}=[{top_items}]"
        )
        logger.debug({
            'method': 'ANOVA_F',
            'selected_count': len(self.selected_features),
            'total_features': len(X.columns),
            'ranked_features': feature_scores
        })
        
        result_df = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        
        return result_df
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted selector."""
        if self.selector is None or self.selected_features is None:
            raise ValueError("Selector not fitted")
        
        available_features = [f for f in self.selected_features if f in X.columns]
        
        if len(available_features) != len(self.selected_features):
            missing_features = [f for f in self.selected_features if f not in X.columns]
            logger.warning(f"feature_mismatch missing={missing_features} available={len(available_features)}")
        
        return X[available_features]

class FeatureScaler:
    """Robust feature scaling."""
    
    def __init__(self):
        self.scaler = RobustScaler()
        
    def fit_transform(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """Fit scaler and transform features using RobustScaler."""
        exclude_cols = exclude_cols or []
        
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        all_exclude_cols = list(set(exclude_cols + datetime_cols))
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.difference(all_exclude_cols)
        
        if len(numerical_cols) == 0:
            return df
        
        df_result = df.copy()
        df_result[numerical_cols] = self.scaler.fit_transform(df_result[numerical_cols])
        
        logger.info(f"feature_scaling applied_to={len(numerical_cols)}_features")
        return df_result
    
    def transform(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """Transform using fitted RobustScaler."""
        exclude_cols = exclude_cols or []
        
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        all_exclude_cols = list(set(exclude_cols + datetime_cols))
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.difference(all_exclude_cols)
        
        df_result = df.copy()
        if len(numerical_cols) > 0:
            df_result[numerical_cols] = self.scaler.transform(df_result[numerical_cols])
        
        return df_result

class FeatureAnalyzer:
    """Feature analysis and diagnostics."""
    
    @staticmethod
    def analyze_features(df: pd.DataFrame, target_col: str = 'availability_target') -> Dict:
        """Basic feature analysis."""
        if target_col not in df.columns:
            return {}
        
        df_clean = df.dropna(subset=[target_col]).copy()
        if df_clean.empty:
            return {}
            
        feature_cols = [col for col in df_clean.columns if col not in [target_col, 'station_id']]
        X = df_clean[feature_cols].copy()
        y = df_clean[target_col]
        
        X = X.dropna(axis=1, how='all')
        
        X = X.dropna()
        y = y.loc[X.index]
        
        if X.empty or len(y) == 0:
            return {}
        
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.Categorical(X[col]).codes
        
        f_scores, _ = f_classif(X, y)
        feature_importance = dict(zip(X.columns, f_scores))
        top_n = 5
        ranked = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        summary = ", ".join([f"{k}={v:.3f}" for k, v in ranked[:top_n]])
        logger.info(f"feature_analysis by=ANOVA_F total={len(X.columns)} top{top_n}=[{summary}]")
        logger.debug({'method': 'ANOVA_F', 'ranked_features': ranked})
        
        return {
            'feature_importance': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True),
            'total_features': len(X.columns)
        }

