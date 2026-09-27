"""
Matching model module using an ensemble of LightGBM and CatBoost for Business Entity Resolution.
Trains binary classification GBDTs on true matches and hard negatives retrieved by blocking.
Licenses:
- LightGBM: MIT License (parameter count << 8B parameters)
- CatBoost: Apache-2.0 License (parameter count << 8B parameters)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import lightgbm as lgb
from catboost import CatBoostClassifier
import numpy as np

_src_dir = Path(__file__).resolve().parent
_code_dir = _src_dir.parents[1]
for _p in [str(_src_dir), str(_code_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from business_entity_resolution.src.features import FEATURE_NAMES, extract_pairwise_features
except ImportError:
    from features import FEATURE_NAMES, extract_pairwise_features


class EntityMatchingModel:
    """
    Ensemble Matching Classifier (LightGBM + CatBoost) for scoring candidate pairs.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        random_state: int = 42,
        ensemble_weight_lgb: float = 0.50,
    ):
        self.params_lgb = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": 20,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "verbose": -1,
            "random_state": random_state,
            "n_estimators": n_estimators,
        }
        self.params_cb = {
            "iterations": n_estimators,
            "learning_rate": learning_rate,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": random_state,
            "verbose": 0,
            "thread_count": -1,
        }
        self.ensemble_weight_lgb = ensemble_weight_lgb
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.lgb_booster: Optional[lgb.Booster] = None
        self.cb_model: Optional[CatBoostClassifier] = None
        self.feature_names = FEATURE_NAMES

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit both LightGBM and CatBoost binary classifiers on feature matrix.
        """
        print(f"Training LightGBM model ({self.params_lgb['n_estimators']} trees, {X_train.shape[0]:,} samples)...", flush=True)
        self.lgb_model = lgb.LGBMClassifier(**self.params_lgb)
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.lgb_model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
        )
        self.lgb_booster = self.lgb_model.booster_

        print(f"Training CatBoost model ({self.params_cb['iterations']} trees, depth={self.params_cb['depth']})...", flush=True)
        try:
            self.cb_model = CatBoostClassifier(**self.params_cb)
            eval_set_cb = (X_val, y_val) if (X_val is not None and y_val is not None) else None
            self.cb_model.fit(
                X_train,
                y_train,
                eval_set=eval_set_cb,
                early_stopping_rounds=30,
            )
            print("CatBoost training completed successfully!", flush=True)
        except Exception as e:
            print(f"[WARNING] CatBoost training encountered error: {e}. Falling back to LightGBM only.", flush=True)
            self.cb_model = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return ensemble predicted match probability for each pair in X.
        """
        if self.lgb_booster is not None:
            p_lgb = self.lgb_booster.predict(X)
            if len(p_lgb.shape) > 1 and p_lgb.shape[1] == 2:
                p_lgb = p_lgb[:, 1]
        elif self.lgb_model is not None:
            p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        else:
            raise ValueError("LightGBM model has not been trained or loaded yet.")

        if self.cb_model is not None:
            p_cb = self.cb_model.predict_proba(X)[:, 1]
            # Weighted average blend
            w = self.ensemble_weight_lgb
            return (w * p_lgb + (1.0 - w) * p_cb).astype(np.float32)

        return p_lgb.astype(np.float32)

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Return mapping of feature names to importance scores.
        """
        if self.lgb_booster is not None:
            imp = self.lgb_booster.feature_importance(importance_type="gain")
            return dict(zip(self.feature_names, imp.tolist()))
        elif self.lgb_model is not None:
            imp = self.lgb_model.feature_importances_
            return dict(zip(self.feature_names, imp.tolist()))
        return {}

    def save(self, model_path: Path) -> None:
        """Save model(s) to file."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Save LightGBM
        if self.lgb_booster is not None:
            self.lgb_booster.save_model(str(model_path))
        elif self.lgb_model is not None:
            self.lgb_model.booster_.save_model(str(model_path))

        # Save CatBoost alongside
        if self.cb_model is not None:
            cb_path = model_path.with_name(model_path.stem + "_catboost.cbm")
            self.cb_model.save_model(str(cb_path))

    def load(self, model_path: Path) -> None:
        """Load model(s) from file."""
        if model_path.is_file():
            self.lgb_booster = lgb.Booster(model_file=str(model_path))

        cb_path = model_path.with_name(model_path.stem + "_catboost.cbm")
        if cb_path.is_file():
            try:
                self.cb_model = CatBoostClassifier()
                self.cb_model.load_model(str(cb_path))
                print(f"Loaded CatBoost model from {cb_path}", flush=True)
            except Exception as e:
                print(f"[INFO] Could not load CatBoost model: {e}. Using LightGBM.", flush=True)
                self.cb_model = None
