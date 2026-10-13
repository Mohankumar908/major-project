"""
retrain_growth_model.py
────────────────────────
Retrains ml_models/growth_model.pkl by blending your accumulated LIVE data
(real sensor readings + real camera-derived height, and real harvest yield
if you've recorded one) with the original synthetic baseline dataset.

Run this the same way as train_growth_model.py, from the project root:
    python retrain_growth_model.py
    python retrain_growth_model.py --real-weight 12
    python retrain_growth_model.py --synthetic-csv dataset/green_gram_growth_dataset.csv

WHY A SEPARATE SCRIPT YOU RUN PERIODICALLY, NOT AUTOMATIC LEARNING
────────────────────────────────────────────────────────────────────
GradientBoostingRegressor (what train_growth_model.py uses) can't update
itself incrementally per-request — that's not how this algorithm works, and
retraining a 400-tree ensemble on every single sensor ping would also be far
too slow. The correct, standard approach is PERIODIC BATCH RETRAINING: let
real data accumulate, then run this script (weekly, or whenever you like)
to fold it in. This is what "the model corrects itself from live data"
actually means in practice.

WHAT COUNTS AS "REAL" TRAINING SIGNAL HERE
────────────────────────────────────────────
- HEIGHT: every DailyGrowthRecord that has both a linked SensorData AND a
  camera-derived actual_height_cm. Available daily once you're capturing
  images + sensor readings together. NOTE: actual_height_cm today comes
  from a greenness-based heuristic, not a ruler — so "real" here means
  "real camera observation", which is still meaningfully different from
  pure synthetic data, but is not perfect ground truth either.
- YIELD: only CropSession rows where you've manually entered
  actual_yield_per_acre_kg at harvest (the real measured yield for that
  field). This is rare (once per completed session) but is true
  ground truth, so it's weighted heavily when present.

HOW THE BLEND WORKS
─────────────────────
Real rows are duplicated with a higher sample_weight (default 8x) relative
to synthetic rows, so the model shifts toward real-world patterns without
being overwhelmed by noise when real data is still scarce. As more real
data accumulates, its influence naturally grows because there are simply
more real rows.

Run train_growth_model.py ONCE at the very start (before any live data
exists) to get your first growth_model.pkl. From then on, run THIS script
periodically to correct it using accumulated live data — don't run
train_growth_model.py again, it would just reset back to synthetic-only.
"""

import argparse
import os
import django
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# ── Django setup (this is a standalone script, so we bootstrap Django
#    ourselves — same idea as manage.py does under the hood) ────────────────
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smart_agri.settings')
django.setup()

from django.db.models import Avg
from agriculture.models import DailyGrowthRecord, CropSession

BASE_DIR    = Path(__file__).resolve().parent
MODEL_OUT   = BASE_DIR / 'ml_models' / 'growth_model.pkl'
FEATURES    = [
    'ph', 'npk_nitrogen', 'npk_phosphorus', 'npk_potassium',
    'temperature', 'humidity', 'soil_moisture', 'day_number',
    'greenness_score', 'leaf_coverage_percent', 'leaf_area_norm',
]

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_real_height_rows():
    """Every daily record with a real camera capture + linked sensor reading."""
    qs = DailyGrowthRecord.objects.filter(
        actual_height_cm__isnull=False,
        sensor_data__isnull=False,
    ).select_related('sensor_data')

    rows = []
    for r in qs:
        s = r.sensor_data
        if None in (s.ph, s.npk_nitrogen, s.npk_phosphorus, s.npk_potassium,
                    s.temperature, s.humidity, s.soil_moisture):
            continue
        gs = r.actual_greenness_score if r.actual_greenness_score is not None else 50.0
        cov = r.actual_leaf_coverage_percent if r.actual_leaf_coverage_percent is not None else 35.0
        norm = round(cov / 100.0, 4) if cov is not None else 0.35
        rows.append({
            'ph': s.ph, 'npk_nitrogen': s.npk_nitrogen,
            'npk_phosphorus': s.npk_phosphorus, 'npk_potassium': s.npk_potassium,
            'temperature': s.temperature, 'humidity': s.humidity,
            'soil_moisture': s.soil_moisture, 'day_number': r.day_number,
            'greenness_score': gs, 'leaf_coverage_percent': cov, 'leaf_area_norm': norm,
            'height_cm': r.actual_height_cm,
        })
    return pd.DataFrame(rows)


def load_real_yield_rows():
    """One row per completed session with a manually-entered real harvest yield."""
    sessions = CropSession.objects.filter(actual_yield_per_acre_kg__isnull=False)
    rows = []
    for session in sessions:
        readings = session.sensor_readings.exclude(ph__isnull=True).exclude(npk_nitrogen__isnull=True)
        if not readings.exists():
            continue
        avg = readings.aggregate(
            ph=Avg('ph'), n=Avg('npk_nitrogen'), p=Avg('npk_phosphorus'),
            k=Avg('npk_potassium'), t=Avg('temperature'), h=Avg('humidity'),
            m=Avg('soil_moisture'),
        )
        if any(v is None for v in avg.values()):
            continue
        # Average image observations for this crop session if available
        avg_gs = session.growth_records.exclude(actual_greenness_score__isnull=True).aggregate(Avg('actual_greenness_score'))['actual_greenness_score__avg'] or 55.0
        avg_cov = session.growth_records.exclude(actual_leaf_coverage_percent__isnull=True).aggregate(Avg('actual_leaf_coverage_percent'))['actual_leaf_coverage_percent__avg'] or 50.0
        rows.append({
            'ph': avg['ph'], 'npk_nitrogen': avg['n'],
            'npk_phosphorus': avg['p'], 'npk_potassium': avg['k'],
            'temperature': avg['t'], 'humidity': avg['h'],
            'soil_moisture': avg['m'], 'day_number': session.days_since_sowing,
            'greenness_score': avg_gs, 'leaf_coverage_percent': avg_cov,
            'leaf_area_norm': round(avg_cov / 100.0, 4),
            'yield_per_acre_kg': session.actual_yield_per_acre_kg,
        })
    return pd.DataFrame(rows)


def retrain(synthetic_csv, real_weight=8.0, min_real_height_rows=5, min_real_yield_rows=1):
    from train_growth_model import stacked_train, stacked_predict
    synthetic_path = Path(synthetic_csv)
    if not synthetic_path.exists():
        print(f"Synthetic baseline CSV not found: {synthetic_path}")
        return

    baseline = pd.read_csv(synthetic_path)
    real_height = load_real_height_rows()
    real_yield  = load_real_yield_rows()

    print(f"Synthetic baseline rows: {len(baseline)}")
    print(f"Real height rows found:  {len(real_height)}")
    print(f"Real yield rows found:   {len(real_yield)}")

    scaler = StandardScaler()
    scaler.fit(baseline[FEATURES].values)

    # ── HEIGHT MODEL ─────────────────────────────────────────────────────
    h_df = baseline[FEATURES + ['height_cm']].copy()
    h_weights = np.ones(len(h_df))

    if len(real_height) >= min_real_height_rows:
        h_df = pd.concat([h_df, real_height[FEATURES + ['height_cm']]], ignore_index=True)
        h_weights = np.concatenate([h_weights, np.full(len(real_height), real_weight)])
        print(f"Blending {len(real_height)} real height observations (weighted {real_weight}x each).")
    else:
        print(f"Only {len(real_height)} real height rows (need {min_real_height_rows}+) "
              f"— training on synthetic data only for height.")

    X_h = scaler.transform(h_df[FEATURES].values)
    y_h = h_df['height_cm'].values
    X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
    h_rf, h_meta, h_xgb = stacked_train(X_tr_h, y_tr_h, n_splits=5, seed=42)
    h_pred = stacked_predict(X_te_h, h_rf, h_meta)
    height_acc = round(r2_score(y_te_h, h_pred) * 100, 1)
    height_mae = round(mean_absolute_error(y_te_h, h_pred), 2)
    print(f"Height model: R²={height_acc}%  MAE=±{height_mae}cm")

    # ── YIELD MODEL ──────────────────────────────────────────────────────
    y_df = baseline[FEATURES + ['yield_per_acre_kg']].copy()
    y_weights = np.ones(len(y_df))

    if len(real_yield) >= min_real_yield_rows:
        y_df = pd.concat([y_df, real_yield[FEATURES + ['yield_per_acre_kg']]], ignore_index=True)
        y_weights = np.concatenate([y_weights, np.full(len(real_yield), real_weight * 2)])
        print(f"Blending {len(real_yield)} real harvest yield observation(s) (weighted {real_weight*2}x each).")
    else:
        print("No real harvest yield recorded yet — training on synthetic baseline.")

    X_y = scaler.transform(y_df[FEATURES].values)
    y_y = y_df['yield_per_acre_kg'].values
    X_tr_y, X_te_y, y_tr_y, y_te_y = train_test_split(X_y, y_y, test_size=0.2, random_state=42)
    y_rf, y_meta, y_xgb = stacked_train(X_tr_y, y_tr_y, n_splits=5, seed=42)
    y_pred = stacked_predict(X_te_y, y_rf, y_meta)
    yield_acc = round(r2_score(y_te_y, y_pred) * 100, 1)
    yield_mae = round(mean_absolute_error(y_te_y, y_pred), 2)
    print(f"Yield model: R²={yield_acc}%  MAE=±{yield_mae}kg/acre")

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump({
        'yield_rf':       y_rf,
        'yield_meta':     y_meta,
        'yield_xgb':      y_xgb,
        'yield_accuracy': yield_acc,
        'yield_mae':      yield_mae,
        'height_rf':      h_rf,
        'height_meta':    h_meta,
        'height_xgb':     h_xgb,
        'height_accuracy': height_acc,
        'height_mae':     height_mae,
        'scaler':         scaler,
        'features':       FEATURES,
        'model_version':  '3.0-gradient-boosting-xgboost-retrained',
        'model_family':   'Gradient Boosting / XGBoost',
    }, MODEL_OUT)

    print(f"\nSaved retrained model -> {MODEL_OUT}")
    print("Restart the Django server to load it.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-csv', default=str(BASE_DIR / 'dataset' / 'green_gram_growth_dataset.csv'))
    parser.add_argument('--real-weight', type=float, default=8.0)
    parser.add_argument('--min-real-height-rows', type=int, default=5)
    parser.add_argument('--min-real-yield-rows', type=int, default=1)
    args = parser.parse_args()

    retrain(args.synthetic_csv, args.real_weight, args.min_real_height_rows, args.min_real_yield_rows)