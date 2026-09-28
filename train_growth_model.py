"""
train_growth_model.py
─────────────────────
Train the Green Gram growth & yield model using a STACKED ENSEMBLE of
RandomForestRegressor (level-0) + GradientBoostingRegressor (level-1 meta).

WHY THIS ENSEMBLE?
──────────────────
  • RandomForest handles noisy, non-linear sensor+image interactions well and
    is robust to outliers (common in IoT data).
  • GradientBoosting as the meta-learner corrects systematic residual errors
    from the Random Forest, squeezing out extra accuracy.
  • Cross-validated out-of-fold (OOF) predictions are used to train the
    meta-learner, preventing data leakage and giving reliable R² estimates.

FEATURES (11):
  Sensor (8): ph, npk_nitrogen, npk_phosphorus, npk_potassium,
              temperature, humidity, soil_moisture, day_number
  Image  (3): greenness_score, leaf_coverage_percent, leaf_area_norm

Usage:
    python train_growth_model.py                    ← auto-generates dataset first
    python train_growth_model.py --csv your.csv     ← use your own CSV
    python train_growth_model.py --no-generate      ← skip dataset generation

Your CSV must have columns:
    ph, npk_nitrogen, npk_phosphorus, npk_potassium,
    temperature, humidity, soil_moisture, day_number,
    greenness_score, leaf_coverage_percent, leaf_area_norm,
    height_cm, yield_per_acre_kg

Output: ml_models/growth_model.pkl  (auto-loaded by Django on next request)
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / 'dataset' / 'green_gram_growth_dataset.csv'
MODEL_OUT   = BASE_DIR / 'ml_models' / 'growth_model.pkl'

FEATURES = [
    'ph', 'npk_nitrogen', 'npk_phosphorus', 'npk_potassium',
    'temperature', 'humidity', 'soil_moisture', 'day_number',
    # image-derived features
    'greenness_score', 'leaf_coverage_percent', 'leaf_area_norm',
]

# ── Stacked ensemble ───────────────────────────────────────────────────────────

def build_rf(seed=42):
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=seed,
    )

def build_gbr_meta(seed=42):
    """Lightweight GBR as meta-learner — trained on OOF predictions."""
    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        subsample=0.8,
        random_state=seed,
    )


def stacked_train(X_tr, y_tr, n_splits=5, seed=42):
    """
    Train level-0 RandomForest with k-fold OOF predictions,
    then fit a GBR meta-learner on those OOF predictions.
    Returns: (rf_model, meta_model) both fitted on full X_tr.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y_tr))

    # Generate OOF predictions for meta-learner training
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
        rf_fold = build_rf(seed + fold)
        rf_fold.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_preds[val_idx] = rf_fold.predict(X_tr[val_idx])

    # Fit meta-learner on OOF
    meta = build_gbr_meta(seed)
    meta.fit(oof_preds.reshape(-1, 1), y_tr)

    # Refit full RF on all training data
    rf_full = build_rf(seed)
    rf_full.fit(X_tr, y_tr)

    return rf_full, meta


def stacked_predict(X, rf_model, meta_model):
    rf_preds = rf_model.predict(X)
    return meta_model.predict(rf_preds.reshape(-1, 1))


# ── Training pipeline ─────────────────────────────────────────────────────────

def train(csv_path: Path):
    print(f"\n{'='*60}")
    print("  Smart Agriculture — Growth Model Training")
    print(f"  Stacked Ensemble: RandomForest → GradientBoosting")
    print(f"  Features: {len(FEATURES)} (sensor + image)")
    print(f"{'='*60}\n")

    print(f"[1/5] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path).dropna(subset=FEATURES + ['height_cm', 'yield_per_acre_kg'])
    print(f"      Rows: {len(df)}")
    print(f"      Columns: {list(df.columns)}\n")

    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    for target_col, key, label, unit in [
        ('yield_per_acre_kg', 'yield', 'Yield  (kg/acre)', 'kg/acre'),
        ('height_cm',         'height', 'Height (cm)',     'cm'),
    ]:
        y = df[target_col].values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        print(f"[2/5] Training {label} stacked ensemble …")

        # Level-0: RandomForest (with OOF) + Level-1: GBR meta
        rf_model, meta_model = stacked_train(X_tr, y_tr, n_splits=5, seed=42)

        # Evaluate stacked predictions on held-out test set
        y_pred_stack = stacked_predict(X_te, rf_model, meta_model)
        r2_stack  = round(r2_score(y_te, y_pred_stack) * 100, 2)
        mae_stack = round(mean_absolute_error(y_te, y_pred_stack), 2)

        # Compare with RF alone (baseline)
        y_pred_rf = rf_model.predict(X_te)
        r2_rf  = round(r2_score(y_te, y_pred_rf) * 100, 2)

        print(f"      RandomForest alone  R²: {r2_rf}%")
        print(f"      Stacked ensemble    R²: {r2_stack}%  MAE: ±{mae_stack} {unit}")
        chosen = r2_stack

        # Feature importance from the RF (interpretable)
        fi = sorted(zip(FEATURES, rf_model.feature_importances_), key=lambda x: -x[1])
        print(f"      Feature importance ({label}):")
        for name, imp in fi:
            bar = '█' * int(imp * 50)
            print(f"        {name:<25} {bar:<25} {round(imp*100, 1)}%")
        print()

        results[key] = {
            'rf':    rf_model,
            'meta':  meta_model,
            'r2':    chosen,
            'mae':   mae_stack,
        }

    print("[3/5] Generating actual vs predicted summary …")
    # Full-dataset predictions for the report
    y_yield_actual = df['yield_per_acre_kg'].values
    y_ht_actual    = df['height_cm'].values
    y_yield_pred   = stacked_predict(X_scaled, results['yield']['rf'], results['yield']['meta'])
    y_ht_pred      = stacked_predict(X_scaled, results['height']['rf'], results['height']['meta'])

    report_df = df[['day_number', 'greenness_score', 'leaf_coverage_percent']].copy()
    report_df['actual_yield']       = np.round(y_yield_actual, 1)
    report_df['predicted_yield']    = np.round(y_yield_pred,   1)
    report_df['yield_error']        = np.round(y_yield_pred - y_yield_actual, 1)
    report_df['actual_height']      = np.round(y_ht_actual, 1)
    report_df['predicted_height']   = np.round(y_ht_pred,   1)
    report_df['height_error']       = np.round(y_ht_pred - y_ht_actual, 1)

    report_path = BASE_DIR / 'dataset' / 'actual_vs_predicted_report.csv'
    report_path.parent.mkdir(exist_ok=True)
    report_df.to_csv(report_path, index=False)
    print(f"      Saved → {report_path}")
    print()

    print("[4/5] Saving model …")
    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump({
        # Yield sub-model
        'yield_rf':       results['yield']['rf'],
        'yield_meta':     results['yield']['meta'],
        'yield_accuracy': results['yield']['r2'],
        'yield_mae':      results['yield']['mae'],
        # Height sub-model
        'height_rf':      results['height']['rf'],
        'height_meta':    results['height']['meta'],
        'height_accuracy': results['height']['r2'],
        'height_mae':     results['height']['mae'],
        # Shared scaler
        'scaler':         scaler,
        'features':       FEATURES,
        'model_version':  '2.0-stacked',
    }, MODEL_OUT)
    print(f"      Saved → {MODEL_OUT}\n")

    print("[5/5] Summary")
    print(f"{'─'*45}")
    print(f"  Yield  R²: {results['yield']['r2']}%   MAE: ±{results['yield']['mae']} kg/acre")
    print(f"  Height R²: {results['height']['r2']}%   MAE: ±{results['height']['mae']} cm")
    print(f"  Features : {len(FEATURES)} (8 sensor + 3 image)")
    print(f"  Ensemble : RandomForest (n=500) → GBR meta (n=200)")
    print(f"{'─'*45}")
    print("\n  ✅ Restart Django server to load the new model.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=str(DEFAULT_CSV), help='Path to training CSV')
    parser.add_argument('--no-generate', action='store_true',
                        help='Skip dataset generation (use existing CSV)')
    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not args.no_generate:
        print("[0/5] Generating synthetic dataset …")
        from generate_growth_dataset import generate
        df_gen = generate(n_rows=3000, seed=42)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_gen.to_csv(csv_path, index=False)
        print(f"      Generated {len(df_gen)} rows → {csv_path}\n")

    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        print("Run without --no-generate to auto-create it.")
        sys.exit(1)

    train(csv_path)
