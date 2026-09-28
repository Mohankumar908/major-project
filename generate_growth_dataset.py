"""
generate_growth_dataset.py
───────────────────────────
Builds a documented, reproducible Green Gram growth & yield dataset that
includes BOTH sensor readings AND image-derived features, for training the
updated growth_model.pkl (see train_growth_model.py).

FEATURES (11 total):
  Sensor (8): ph, npk_nitrogen, npk_phosphorus, npk_potassium,
              temperature, humidity, soil_moisture, day_number
  Image  (3): greenness_score      – Excess Green Index (0–100)
              leaf_coverage_percent – HSV-mask coverage fraction (0–100)
              leaf_area_norm        – leaf area normalised [0–1] (dimensionless
                                     proxy when no camera calibration exists)

WHY SYNTHETIC IMAGE FEATURES?
──────────────────────────────
Real greenness / leaf-coverage values are extracted from photos by
ml_utils.extract_greenness() and extract_leaf_area().  During initial
training (before live images accumulate) we synthesise plausible values
from the same agronomic model that drives height:
  - Greenness follows the same S-curve as height but plateaus earlier
    (leaf colour peaks at vegetative stage, drops at senescence).
  - Leaf coverage mirrors canopy expansion, modulated by temperature and
    moisture stress.
  - Leaf area norm = leaf_coverage / 100 (dimensionless, same as coverage
    percent but scaled to [0,1] for model input).

Once real images arrive, retrain_growth_model.py blends them in with high
sample weight, overriding the synthetic proxies.

WHAT'S REAL VS. ASSUMED — state this in your report:
  • Input feature RANGES are consistent with Indian smallholder conditions.
  • HEIGHT and YIELD formulas: documented synthetic approximation (see below).
  • IMAGE features: synthesised from the same agronomic model — explicitly
    NOT observed pixel values. Their purpose is to give the model the right
    feature dimensionality at cold start so no code change is needed when
    real image data arrives.

Usage:
    python generate_growth_dataset.py --rows 3000
    python generate_growth_dataset.py --rows 3000 --out dataset/green_gram_growth_dataset.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def generate(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    # ── Sensor features ─────────────────────────────────────────────────────
    ph          = rng.uniform(5.2, 8.0, n_rows)
    nitrogen    = rng.uniform(15, 130, n_rows)
    phosphorus  = rng.uniform(8, 85, n_rows)
    potassium   = rng.uniform(15, 105, n_rows)
    temperature = rng.uniform(16, 40, n_rows)
    humidity    = rng.uniform(35, 92, n_rows)
    moisture    = rng.uniform(15, 85, n_rows)
    day_number  = rng.uniform(1, 65, n_rows)

    # ── Height (cm): logistic S-curve ───────────────────────────────────────
    base_height = 4 + 46 * (1 / (1 + np.exp(-0.13 * (day_number - 26))))
    height_cm = (
        base_height
        + 2.5 * (nitrogen / 100)
        - 3.0 * np.abs(ph - 6.7)
        - 0.05 * np.maximum(0, 33 - temperature)
        - 0.04 * np.maximum(0, 30 - moisture)
        + rng.normal(0, 1.6, n_rows)
    )
    height_cm = np.clip(height_cm, 2, 55)

    # ── Greenness score (0–100): peaks mid-season, falls at senescence ───────
    # Greeness peaks at vegetative/flowering stage (~day 25-35) then drops
    # as pods mature and leaves yellow — shaped as a skewed bell.
    greenness_base = (
        20 + 65 * np.exp(-0.5 * ((day_number - 30) / 18) ** 2)  # bell centred day 30
        + 4.0 * (moisture / 80)                                  # moisture boost
        - 5.0 * np.abs(ph - 6.7)                                 # pH penalty
        - 0.3 * np.maximum(0, temperature - 33)                  # heat stress
        + rng.normal(0, 4.0, n_rows)
    )
    greenness_score = np.clip(greenness_base, 5, 95)

    # ── Leaf coverage percent (0–100): mirrors canopy expansion ─────────────
    leaf_cov_base = (
        3 + 70 * (1 / (1 + np.exp(-0.11 * (day_number - 22))))  # S-curve, slightly earlier
        - 8.0 * np.abs(ph - 6.7)
        - 0.08 * np.maximum(0, 30 - moisture)
        + 0.3 * (nitrogen / 100)
        + rng.normal(0, 4.5, n_rows)
    )
    leaf_coverage_percent = np.clip(leaf_cov_base, 2, 92)

    # ── Leaf area norm [0–1] ────────────────────────────────────────────────
    leaf_area_norm = np.clip(leaf_coverage_percent / 100.0, 0.02, 0.92)

    # ── Yield (kg/acre): linear approximation ───────────────────────────────
    yield_per_acre_kg = (
        300
        + 1.0 * nitrogen
        + 0.8 * phosphorus
        + 0.6 * potassium
        - 12  * np.abs(ph - 6.7)
        + 0.3 * humidity
        - 3.0 * np.abs(temperature - 28)
        + 0.6 * moisture
        + 2.0 * day_number
        + 15  * (greenness_score / 100)   # greener canopy → higher yield
        + 20  * (leaf_coverage_percent / 100)  # denser canopy → higher yield
        + rng.normal(0, 14, n_rows)
    )
    yield_per_acre_kg = np.clip(yield_per_acre_kg, 100, 850)

    df = pd.DataFrame({
        'ph':                    np.round(ph, 2),
        'npk_nitrogen':          np.round(nitrogen, 1),
        'npk_phosphorus':        np.round(phosphorus, 1),
        'npk_potassium':         np.round(potassium, 1),
        'temperature':           np.round(temperature, 1),
        'humidity':              np.round(humidity, 1),
        'soil_moisture':         np.round(moisture, 1),
        'day_number':            np.round(day_number).astype(int),
        'greenness_score':       np.round(greenness_score, 2),
        'leaf_coverage_percent': np.round(leaf_coverage_percent, 2),
        'leaf_area_norm':        np.round(leaf_area_norm, 4),
        'height_cm':             np.round(height_cm, 1),
        'yield_per_acre_kg':     np.round(yield_per_acre_kg, 1),
    })
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=str(BASE_DIR / 'dataset' / 'green_gram_growth_dataset.csv'))
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} rows  →  {out_path}")
    print(f"\nFeatures: {list(df.columns)}")
    print("\nColumn ranges:")
    print(df.describe().round(1))
    print(
        "\nIMPORTANT: image features (greenness_score, leaf_coverage_percent, "
        "leaf_area_norm) are SYNTHETIC here. They will be overridden by real "
        "camera-derived values when you retrain with retrain_growth_model.py."
    )
