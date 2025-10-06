#!/usr/bin/env python3
"""
train.py — SageMaker-compatible training entrypoint (clean version)

Behaviour:
- read hyperparameters from /opt/ml/input/config/hyperparameters.json
- load training data from /opt/ml/input/data/training
- train model
- save model ONNX file to /opt/ml/model (SageMaker uploads it automatically)
"""

import os
import sys
import json
import argparse
import logging
import warnings
import io
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- Utility functions ----------

def read_sagemaker_hyperparameters():
    """Load hyperparameters passed by SageMaker."""
    hp_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(hp_path):
        with open(hp_path, "r") as f:
            raw = json.load(f)
        return raw
    return {}


def weighted_rmse(y_true, y_pred, tindex, power=2):
    weights = tindex ** power
    mse = np.average((y_true - y_pred) ** 2, weights=weights)
    return np.sqrt(mse)


def convert_model_to_onnx_bytes(model, df, feature_cols):
    """Convert trained model to ONNX bytes."""
    X_sample = df[feature_cols].values[:1]
    initial_type = [("input", FloatTensorType([None, X_sample.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    return onnx_model.SerializeToString()


def find_local_training_csv():
    """Return path to first CSV under SageMaker training channel."""
    local_dir = Path("/opt/ml/input/data/training")
    if local_dir.exists():
        csvs = list(local_dir.glob("**/*.csv"))
        if csvs:
            return str(csvs[0])
    return None


# ---------- Model training ----------

def train_model(df, feature_cols, n_splits=5, power=2):
    """Train stacking ensemble with GroupKFold and return final model."""
    target_col = 'rul_percentage'

    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['round'].values
    tindex_raw = df['index'].values

    # global normalization of tindex
    tindex_min = tindex_raw.min()
    tindex_max = tindex_raw.max()
    tindex_full = (tindex_raw - tindex_min) / (tindex_max - tindex_min + 1e-9)

    if n_splits > len(np.unique(groups)):
        n_splits = len(np.unique(groups))
        logger.info(f"Adjusted n_splits to {n_splits} to match unique rounds.")

    outer_gkf = GroupKFold(n_splits=n_splits)

    estimators = [
        ('lgbm', LGBMRegressor(random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=0)),
        ('xgb', XGBRegressor(random_state=42, verbosity=0))
    ]

    try:
        import sklearn
        from sklearn import set_config
        ver = tuple(int(x) for x in sklearn.__version__.split('.')[:2])
        if ver >= (1, 6):
            set_config(enable_metadata_routing=True)
    except Exception:
        warnings.warn("Could not configure sklearn metadata routing")

    rmses, wrmses, reliability_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(outer_gkf.split(X, y, groups=groups), 1):
        logger.info(f"=== Outer Fold {fold}/{n_splits} ===")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        tindex_train = tindex_full[train_idx]
        tindex_val = tindex_full[val_idx]
        sample_weights_train = tindex_train ** power

        inner_splits = max(2, min(n_splits, len(np.unique(groups[train_idx]))))
        gkf_inner = GroupKFold(n_splits=inner_splits)

        final_estimator = RidgeCV(cv=gkf_inner)
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=gkf_inner,
            passthrough=False,
            n_jobs=-1
        )

        try:
            stacking.fit(X_train, y_train, sample_weight=sample_weights_train, groups=groups[train_idx])
        except TypeError:
            stacking.fit(X_train, y_train, sample_weight=sample_weights_train)

        y_pred = stacking.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        wrmse_val = weighted_rmse(y_val, y_pred, tindex_val, power)

        # simple reliability score from base estimators
        base_preds = []
        for name, _ in estimators:
            base_est = stacking.named_estimators_.get(name)
            if base_est is not None:
                base_preds.append(base_est.predict(X_val))
        if base_preds:
            std_per_sample = np.std(np.column_stack(base_preds), axis=1)
            reliability_score = float(np.mean(np.exp(-std_per_sample)))
        else:
            reliability_score = float('nan')

        rmses.append(rmse)
        wrmses.append(wrmse_val)
        reliability_scores.append(reliability_score)

        logger.info(f"Fold {fold}: RMSE={rmse:.4f} wRMSE={wrmse_val:.4f} reliability={reliability_score:.4f}")

    logger.info("==== Cross-validation summary ====")
    logger.info(f"Avg RMSE: {np.mean(rmses):.4f} Avg wRMSE: {np.mean(wrmses):.4f} Avg reliability: {np.nanmean(reliability_scores):.4f}")

    # Final refit on full data
    final_inner_splits = max(2, min(n_splits, len(np.unique(groups))))
    final_gkf_inner = GroupKFold(n_splits=final_inner_splits)
    final_estimator_full = RidgeCV(cv=final_gkf_inner)
    final_stacking = StackingRegressor(estimators=estimators, final_estimator=final_estimator_full,
                                       cv=final_gkf_inner, passthrough=False, n_jobs=-1)
    try:
        final_stacking.fit(X, y, sample_weight=tindex_full ** power, groups=groups)
    except TypeError:
        final_stacking.fit(X, y, sample_weight=tindex_full ** power)

    return final_stacking, np.mean(rmses), np.mean(wrmses), np.nanmean(reliability_scores)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cols", default=None, help="comma-separated feature columns")
    parser.add_argument("--model-filename", default=None)
    args = parser.parse_args()

    # Override with SageMaker hyperparameters
    sm_hps = read_sagemaker_hyperparameters()
    for keb, val in sm_hps.items():
        key = keb.replace('-', '_')
        if hasattr(args, key):
            setattr(args, key, val)

    # feature_cols required
    if not args.feature_cols:
        logger.error("feature_cols must be provided (comma-separated) via hyperparameters or CLI")
        sys.exit(2)
    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    # Load training data from SageMaker channel
    local_csv = find_local_training_csv()
    if not local_csv:
        logger.error("No training data found under /opt/ml/input/data/training")
        sys.exit(2)
    logger.info(f"Loading training CSV from: {local_csv}")
    df = pd.read_csv(local_csv)

    # Train
    model, mean_rmse, mean_wrmse, mean_reliability = train_model(df, feature_cols=feature_cols)

    # Save model to /opt/ml/model
    model_filename = args.model_filename 
    local_model_path = f"/opt/ml/model/{model_filename}"
    try:
        logger.info("Converting model to ONNX...")
        onnx_bytes = convert_model_to_onnx_bytes(model, df, feature_cols)
        Path(local_model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_model_path, "wb") as f:
            f.write(onnx_bytes) 
        logger.info(f"Saved ONNX model to {local_model_path}")
    except Exception as e:
        logger.exception(f"ONNX conversion failed: {e}. Falling back to pickle.")
        import pickle
        local_pickle = f"/opt/ml/model/{os.path.splitext(model_filename)[0]}.pkl"
        with open(local_pickle, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model pickle to {local_pickle}")

    logger.info(f"Training finished. mean_rmse={mean_rmse:.4f} mean_wrmse={mean_wrmse:.4f} mean_reliability={mean_reliability:.4f}")


if __name__ == "__main__":
    main()
