"""
train.py — Nigerian SME Monthly Sales Prediction
================================================
Production-ready PyCaret training pipeline.
Automatically detects regression vs classification,
compares all models, tunes the best one, and saves it.

Usage:
    python train.py
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = "data/dataset.csv"
MODEL_DIR   = "model"
MODEL_NAME  = "best_model"
TARGET_COL  = None        # None = auto-detect (last column)
SESSION_ID  = 42          # For reproducibility
N_MODELS    = 5           # Top-N models to compare before tuning


def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset and return a DataFrame."""
    print(f"\n Loading dataset from: {path}")
    if not os.path.exists(path):
        sys.exit(f" File not found: {path}")

    df = pd.read_csv(path)
    print(f" Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print basic dataset statistics."""
    print("\n Dataset Overview:")
    print(df.head(3).to_string())
    print(f"\nShape  : {df.shape}")
    print(f"Dtypes :\n{df.dtypes.to_string()}")
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n  Missing values:\n{missing[missing > 0].to_string()}")
    else:
        print("\n No missing values found.")


def detect_task(df: pd.DataFrame, target: str) -> str:
    """
    Detect whether the problem is regression or classification.
    Numeric target with many unique values → regression.
    Categorical or low-cardinality numeric → classification.
    """
    col = df[target]
    if col.dtype == "object" or col.nunique() <= 10:
        return "classification"
    return "regression"


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple missing-value strategy:
      - Numeric  → median imputation
      - Categorical → mode imputation
    PyCaret also handles this internally, but explicit handling is good practice.
    """
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    return df


def train_regression(df: pd.DataFrame, target: str) -> None:
    """Full PyCaret regression pipeline."""
    from pycaret.regression import (
        setup, compare_models, tune_model,
        finalize_model, save_model, pull
    )

    print("\n🔧 Setting up PyCaret Regression experiment …")
    setup(
        data            = df,
        target          = target,
        session_id      = SESSION_ID,
        normalize       = True,          # Scale numeric features
        transformation  = True,          # Power-transform skewed features
        remove_outliers = False,         # Keep all SME data intact
        verbose         = False,
    )

    print("\n🏁 Comparing all regression models …")
    best_models = compare_models(n_select=N_MODELS, verbose=True)

    # compare_models returns a list when n_select > 1
    top_model = best_models[0] if isinstance(best_models, list) else best_models
    print(f"\n Best base model: {type(top_model).__name__}")

    print("\n  Tuning hyperparameters …")
    tuned_model = tune_model(
        top_model,
        optimize   = "R2",    # Primary metric for regression
        n_iter     = 50,      # Bayesian search iterations
        verbose    = False,
    )

    print("\n Tuned model results:")
    results = pull()
    print(results.to_string())

    print("\n Finalizing model on full dataset …")
    final_model = finalize_model(tuned_model)

    # ── Save ──────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    save_model(final_model, save_path)
    print(f"\n Model saved to: {save_path}.pkl")
    print(f"\n Training complete! Best model: {type(final_model).__name__}")


def train_classification(df: pd.DataFrame, target: str) -> None:
    """Full PyCaret classification pipeline."""
    from pycaret.classification import (
        setup, compare_models, tune_model,
        finalize_model, save_model, pull
    )

    print("\n Setting up PyCaret Classification experiment …")
    setup(
        data           = df,
        target         = target,
        session_id     = SESSION_ID,
        normalize      = True,
        verbose        = False,
    )

    print("\n Comparing all classification models …")
    best_models = compare_models(n_select=N_MODELS, verbose=True)

    top_model = best_models[0] if isinstance(best_models, list) else best_models
    print(f"\n Best base model: {type(top_model).__name__}")

    print("\  Tuning hyperparameters …")
    tuned_model = tune_model(
        top_model,
        optimize   = "Accuracy",
        n_iter     = 50,
        verbose    = False,
    )

    print("\n Tuned model results:")
    results = pull()
    print(results.to_string())

    print("\n🔒 Finalizing model on full dataset …")
    final_model = finalize_model(tuned_model)

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    save_model(final_model, save_path)
    print(f"\n Model saved to: {save_path}.pkl")
    print(f"\n Training complete! Best model: {type(final_model).__name__}")


def main():
    print("=" * 60)
    print("   NIGERIAN SME MONTHLY SALES — ML TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Inspect
    inspect_data(df)

    # 3. Auto-detect target column (last column if not specified)
    target = TARGET_COL if TARGET_COL else df.columns[-1]
    print(f"\n Target column: '{target}'")

    # 4. Handle missing values
    df = handle_missing(df)

    # 5. Detect task type
    task = detect_task(df, target)
    print(f" Detected task type: {task.upper()}")

    # 6. Train
    if task == "regression":
        train_regression(df, target)
    else:
        train_classification(df, target)


if __name__ == "__main__":
    main()
