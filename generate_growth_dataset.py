"""
generate_growth_dataset.py
───────────────────────────
Builds a documented, reproducible Green Gram growth & yield dataset for
training growth_model.pkl (see train_growth_model.py).

WHY THIS EXISTS
────────────────
No public dataset combines day-wise Green Gram height with NPK/soil-moisture
readings and yield outcomes (we checked: Kaggle's "crops-npk-data-set",
"crop-yield-and-environmental-factors-2014-2023", "smart-farming-sensor-data",
and rishabhrathore055/datas all lack at least one of {NPK, day-by-day height,
Green Gram}). Rather than force-fit a mismatched dataset, this script
generates a dataset from DOCUMENTED agronomic relationships, partly informed
by the ranges/relationships seen in the public datasets below, and clearly
marks every relationship as either (a) sourced, or (b) a stated assumption.

WHAT'S REAL VS. ASSUMED — be upfront about this in your report/paper:
  • Input feature RANGES (pH, N, P, K, temp, humidity, moisture) are set to
    match realistic Indian smallholder-farm agronomic conditions, consistent
    with the ranges reported in:
      - "crops-npk-data-set" (Kaggle, javakhan) — pH ~4.5–9.35, N/P/K ranges
      - "crop-yield-and-environmental-factors-2014-2023" (Kaggle,
        madhankumar789) — itself a synthetic dataset generated from
        real-world agricultural principles; used here only as a sanity
        check on plausible NPK -> yield sensitivity magnitudes
      - rishabhrathore055/datas — confirms Green Gram (mungbean)'s typical
        pH/temperature/humidity growing envelope
  • The HEIGHT-OVER-TIME curve is a logistic growth model, standard in crop
    growth modelling literature (S-curve: slow early growth, fast mid-season,
    plateau near maturity), parameterised for Green Gram's ~55-65 day cycle
    and ~45-55cm mature height (ICAR pulse crop package-of-practices figures).
  • The exact yield formula (coefficients per unit of N/P/K/pH-deviation/etc.)
    is OUR OWN linear approximation, not copied from a specific paper. This
    is the assumption you should state plainly: "yield sensitivity
    coefficients were approximated by the project team based on documented
    ranges, not fitted to a specific published dataset."
  • Every one of these rows is later meant to be VALIDATED, not just trusted,
    against your own live IoT + camera data collected during actual
    cultivation — that comparison is what makes the overall claim credible,
    not the synthetic training data on its own.

Usage:
    python generate_growth_dataset.py --rows 2000
    python generate_growth_dataset.py --rows 2000 --out dataset/green_gram_growth_dataset.csv --seed 7
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def generate(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    # ── Feature ranges: realistic smallholder-farm agronomic conditions ────
    ph          = rng.uniform(5.2, 8.0, n_rows)
    nitrogen    = rng.uniform(15, 130, n_rows)     # kg/ha-equivalent, legume needs less N (fixes own)
    phosphorus  = rng.uniform(8, 85, n_rows)
    potassium   = rng.uniform(15, 105, n_rows)
    temperature = rng.uniform(16, 40, n_rows)      # °C
    humidity    = rng.uniform(35, 92, n_rows)      # %
    moisture    = rng.uniform(15, 85, n_rows)      # soil moisture %
    day_number  = rng.uniform(1, 65, n_rows)       # Green Gram cycle ~55-65 days

    # ── Height (cm): logistic growth curve ──────────────────────────────────
    # ASSUMPTION: standard S-curve crop growth model. Midpoint ~day 26,
    # mature height ~48cm (+/- nutrient/pH modifiers), per ICAR pulse crop
    # package-of-practices typical figures for Vigna radiata.
    base_height = 4 + 46 * (1 / (1 + np.exp(-0.13 * (day_number - 26))))
    height_cm = (
        base_height
        + 2.5 * (nitrogen / 100)              # mild boost from available N
        - 3.0 * np.abs(ph - 6.7)              # penalise pH far from optimum (~6.2-7.2)
        - 0.05 * np.maximum(0, 33 - temperature)   # cold stress below ~33C threshold (mild)
        - 0.04 * np.maximum(0, 30 - moisture)      # drought stress below ~30% moisture
        + rng.normal(0, 1.6, n_rows)          # measurement/field noise
    )
    height_cm = np.clip(height_cm, 2, 55)

    # ── Yield (kg/acre): OUR linear approximation, not fitted to a specific
    # published dataset. Coefficients scaled so the realistic input ranges
    # above map onto Green Gram's typical 300-700 kg/acre yield band without
    # saturating the clip bounds for most rows (checked below).
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
        + rng.normal(0, 14, n_rows)
    )
    yield_per_acre_kg = np.clip(yield_per_acre_kg, 100, 800)

    df = pd.DataFrame({
        'ph':               np.round(ph, 2),
        'npk_nitrogen':     np.round(nitrogen, 1),
        'npk_phosphorus':   np.round(phosphorus, 1),
        'npk_potassium':    np.round(potassium, 1),
        'temperature':      np.round(temperature, 1),
        'humidity':         np.round(humidity, 1),
        'soil_moisture':    np.round(moisture, 1),
        'day_number':       np.round(day_number).astype(int),
        'height_cm':        np.round(height_cm, 1),
        'yield_per_acre_kg': np.round(yield_per_acre_kg, 1),
    })
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=str(BASE_DIR / 'dataset' / 'green_gram_growth_dataset.csv'))
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} rows -> {out_path}")
    print("\nColumn ranges:")
    print(df.describe().round(1))
    print(
        "\nIMPORTANT: this is a documented synthetic dataset (see docstring "
        "for exactly what's sourced vs. assumed). State this honestly in "
        "your report — do not cite it as a specific downloaded Kaggle "
        "dataset. Validate the trained model against your own live IoT + "
        "camera data before trusting its predictions."
    )
