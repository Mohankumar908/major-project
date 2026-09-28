"""
ml_utils.py
───────────
All ML helpers for the Smart Agriculture platform.

Growth model  (v2 stacked ensemble):
  - 11 features: 8 sensor + 3 image-derived
  - Level-0: RandomForestRegressor (n=500)
  - Level-1 meta: GradientBoostingRegressor (n=200), trained on OOF predictions
  - Two parallel sub-models: yield_per_acre  and  height_cm
  - Falls back to train-on-the-fly if pkl doesn't exist yet

Image helpers:
  - extract_greenness()     → Excess Green Index score 0–100
  - extract_leaf_area()     → HSV-mask leaf coverage % + optional cm²
  - extract_image_features()→ merges both into a dict ready for predict_growth()

Disease model:
  - ResNet-18 CNN  (disease_model.pth)
  - predict_disease() → (diseased_bool, name, confidence, accuracy, treatment, low_conf)
"""

import numpy as np
import os
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'ml_models'
GROWTH_MODEL_PATH = MODEL_DIR / 'growth_model.pkl'

FEATURES = [
    'ph', 'npk_nitrogen', 'npk_phosphorus', 'npk_potassium',
    'temperature', 'humidity', 'soil_moisture', 'day_number',
    'greenness_score', 'leaf_coverage_percent', 'leaf_area_norm',
]

# Default image-feature values used when no image is available
# (sensor-only path). Chosen to be neutral/mid-range so they don't
# skew the model; the sensor features dominate in that case.
_IMG_DEFAULTS = {
    'greenness_score':       55.0,   # mid-range ExG
    'leaf_coverage_percent': 40.0,   # moderate canopy
    'leaf_area_norm':         0.40,  # = coverage/100
}


# ─── GROWTH MODEL LOAD / TRAIN ─────────────────────────────────────────────────

def _train_and_save_growth_model():
    """
    Cold-start: generate a synthetic dataset and train the stacked ensemble.
    Called automatically if growth_model.pkl doesn't exist yet.
    """
    print("[ML] growth_model.pkl not found — training from scratch …")
    import subprocess, sys
    script = BASE_DIR / 'train_growth_model.py'
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BASE_DIR),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("train_growth_model.py failed — check console output.")


def load_growth_model():
    """Load the stacked ensemble dict from pkl. Trains if missing."""
    import joblib
    if not GROWTH_MODEL_PATH.exists():
        _train_and_save_growth_model()
    d = joblib.load(GROWTH_MODEL_PATH)
    # Backwards-compatible with old single-model pkl (pre v2)
    if 'yield_rf' not in d:
        print("[ML] Old pkl format detected — retraining with new ensemble …")
        _train_and_save_growth_model()
        d = joblib.load(GROWTH_MODEL_PATH)
    return d


def _stacked_predict(X, rf_model, meta_model):
    """Run stacked ensemble inference: RF → meta GBR."""
    rf_preds = rf_model.predict(X)
    return meta_model.predict(rf_preds.reshape(-1, 1))


# ─── MAIN PREDICTION ENTRY POINT ──────────────────────────────────────────────

def predict_growth(
    ph, nitrogen, phosphorus, potassium,
    temperature, humidity, moisture,
    day_number=30, field_acres=1.0,
    # image-derived features (optional — defaults used when no image)
    greenness_score=None,
    leaf_coverage_percent=None,
    leaf_area_norm=None,
):
    """
    Predict crop growth using sensor + image features.

    Parameters
    ----------
    ph, nitrogen, phosphorus, potassium : sensor readings
    temperature, humidity, moisture      : sensor readings
    day_number                           : days since sowing
    field_acres                          : field size for dose calculations
    greenness_score                      : 0–100 from extract_greenness()
    leaf_coverage_percent                : 0–100 from extract_leaf_area()
    leaf_area_norm                       : leaf_coverage_percent / 100

    Returns dict with:
        predicted_yield_per_acre, predicted_height_cm,
        health_score, growth_stage, model_accuracy, height_accuracy,
        recommendations (list of dicts), recommendation_text (str),
        image_features_used (bool)
    """
    d = load_growth_model()

    # Fill image feature defaults when no image was uploaded
    gs  = greenness_score        if greenness_score        is not None else _IMG_DEFAULTS['greenness_score']
    lcp = leaf_coverage_percent  if leaf_coverage_percent  is not None else _IMG_DEFAULTS['leaf_coverage_percent']
    lan = leaf_area_norm         if leaf_area_norm         is not None else (lcp / 100.0)
    image_features_used = greenness_score is not None

    X = np.array([[
        ph, nitrogen, phosphorus, potassium,
        temperature, humidity, moisture, day_number,
        gs, lcp, lan,
    ]])
    X_s = d['scaler'].transform(X)

    yield_per_acre = round(float(_stacked_predict(X_s, d['yield_rf'],  d['yield_meta'])[0]),  1)
    pred_height    = round(float(_stacked_predict(X_s, d['height_rf'], d['height_meta'])[0]), 1)

    yield_acc  = d.get('yield_accuracy',  0)
    height_acc = d.get('height_accuracy', 0)

    # ── Health score (rule-based penalty system) ────────────────────────────
    score = 100
    # pH
    if   ph < 5.5 or ph > 7.5:   score -= 25
    elif ph < 6.0 or ph > 7.0:   score -= 10
    # Macronutrients
    if   nitrogen < 30:           score -= 20
    elif nitrogen < 50:           score -= 8
    if   phosphorus < 15:         score -= 12
    if   potassium < 25:          score -= 10
    # Temperature
    if   temperature > 35:        score -= 18
    elif temperature < 18:        score -= 15
    # Moisture / humidity
    if   humidity < 40:           score -= 12
    if   moisture < 25:           score -= 18
    elif moisture < 35:           score -= 8
    # Image-based penalties (when real image data is present)
    if image_features_used:
        if   gs < 20:             score -= 15   # yellowing / disease visible
        elif gs < 35:             score -= 7
        if   lcp < 15:            score -= 10   # very sparse canopy
        elif lcp < 25:            score -= 4
    health_score = max(0, min(100, score))

    # ── Growth stage by day (Green Gram 55-65 day crop) ────────────────────
    if   day_number <= 5:   stage = "Germination"
    elif day_number <= 15:  stage = "Seedling"
    elif day_number <= 30:  stage = "Vegetative"
    elif day_number <= 42:  stage = "Flowering"
    elif day_number <= 55:  stage = "Pod Formation"
    elif day_number <= 65:  stage = "Maturity"
    else:                   stage = "Harvest Ready"

    # ── Detailed recommendations with quantities ────────────────────────────
    recs = []

    if ph < 5.8:
        lime_kg = round((6.5 - ph) * 200 * field_acres, 1)
        recs.append({"type":"soil","icon":"🪨","issue":f"Low pH ({ph})","action":"Apply Agricultural Lime","quantity":f"{lime_kg} kg/acre","timing":"Apply before next irrigation"})
    if ph > 7.2:
        sulfur_kg = round((ph - 6.5) * 120 * field_acres, 1)
        recs.append({"type":"soil","icon":"⚗️","issue":f"High pH ({ph})","action":"Apply Elemental Sulfur","quantity":f"{sulfur_kg} kg/acre","timing":"Mix into top 10 cm soil"})
    if nitrogen < 30:
        urea_kg = round((80 - nitrogen) * 2.17 * field_acres, 1)
        recs.append({"type":"fertilizer","icon":"🌿","issue":f"Nitrogen deficiency ({nitrogen} mg/kg)","action":"Apply Urea (46% N)","quantity":f"{urea_kg} kg/acre","timing":"Split into 2 doses — now & 15 days later"})
    elif nitrogen < 50:
        urea_kg = round((60 - nitrogen) * 2.17 * field_acres, 1)
        recs.append({"type":"fertilizer","icon":"🌿","issue":f"Low Nitrogen ({nitrogen} mg/kg)","action":"Apply Urea (46% N)","quantity":f"{urea_kg} kg/acre","timing":"Apply within 3 days"})
    if phosphorus < 15:
        dap_kg = round((40 - phosphorus) * 5.43 * field_acres, 1)
        recs.append({"type":"fertilizer","icon":"🟤","issue":f"Phosphorus deficiency ({phosphorus} mg/kg)","action":"Apply DAP (18% N, 46% P₂O₅)","quantity":f"{dap_kg} kg/acre","timing":"Basal application before sowing"})
    if potassium < 25:
        mop_kg = round((60 - potassium) * 1.67 * field_acres, 1)
        recs.append({"type":"fertilizer","icon":"🔴","issue":f"Potassium deficiency ({potassium} mg/kg)","action":"Apply MOP (Muriate of Potash, 60% K₂O)","quantity":f"{mop_kg} kg/acre","timing":"Apply at tillering stage"})
    if moisture < 25:
        water_mm = round((45 - moisture) * 3.5 * field_acres, 0)
        recs.append({"type":"water","icon":"💧","issue":f"Critically low soil moisture ({moisture}%)","action":"Irrigate immediately","quantity":f"{int(water_mm)} litres/acre","timing":"Within 24 hours — crop stress risk"})
    elif moisture < 35:
        recs.append({"type":"water","icon":"💧","issue":f"Low soil moisture ({moisture}%)","action":"Schedule irrigation","quantity":f"{int(round((40 - moisture) * 2.5 * field_acres, 0))} litres/acre","timing":"Within 2–3 days"})
    if temperature > 35:
        recs.append({"type":"environment","icon":"🌡️","issue":f"Heat stress ({temperature}°C)","action":"Foliar spray with 1% Potassium Chloride solution","quantity":f"{round(2.5 * field_acres, 1)} litres solution/acre","timing":"Spray in early morning or evening"})
    if humidity < 40:
        recs.append({"type":"environment","icon":"🌫️","issue":f"Low humidity ({humidity}%)","action":"Increase irrigation frequency","quantity":"Light irrigation every 2 days","timing":"Continue until humidity > 55%"})

    # Image-derived recommendations
    if image_features_used:
        if gs < 25:
            recs.append({"type":"image","icon":"🍃","issue":f"Very low greenness ({gs:.1f}%) — possible yellowing or disease","action":"Inspect leaves for disease/pest; consider foliar nutrient spray","quantity":"Foliar Urea 1% solution: 5 litres/acre","timing":"Spray within 24 hours"})
        elif gs < 40:
            recs.append({"type":"image","icon":"🍃","issue":f"Low greenness ({gs:.1f}%) — possible stress","action":"Check nitrogen levels and irrigation","quantity":"Assess current soil moisture","timing":"Monitor over next 2 days"})
        if lcp < 15:
            recs.append({"type":"image","icon":"🌿","issue":f"Sparse canopy ({lcp:.1f}% coverage)","action":"Check germination rate and plant spacing; consider gap filling","quantity":"—","timing":"Assess within 3 days"})

    if not recs:
        recs.append({"type":"ok","icon":"✅","issue":"All conditions optimal","action":"Maintain current practices","quantity":"—","timing":"Monitor sensors daily"})

    return {
        "predicted_yield_per_acre": yield_per_acre,
        "predicted_height_cm":      pred_height,
        "health_score":             round(health_score, 1),
        "growth_stage":             stage,
        "model_accuracy":           yield_acc,
        "height_accuracy":          height_acc,
        "model_version":            d.get('model_version', '1.0'),
        "image_features_used":      image_features_used,
        "recommendations":          recs,
        "recommendation_text":      " | ".join(r['action'] + ": " + r['quantity'] for r in recs),
    }


# ─── IMAGE FEATURE EXTRACTION ──────────────────────────────────────────────────

def extract_greenness(image_path):
    """
    Excess Green Index (ExG) score 0–100.
    ExG = 2G - R - B per pixel; higher = greener / healthier canopy.
    """
    from PIL import Image

    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((128, 128))
        pixels = np.array(img).astype(float)
        r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        exg = 2 * g - r - b
        score = np.clip((np.mean(exg) / 255) * 100, 0, 100)
        return round(float(score), 1)
    except Exception:
        return 55.0   # neutral default


def extract_leaf_area(image_path, frame_area_cm2=None):
    """
    HSV-based green-pixel mask.

    Returns
    -------
    (coverage_percent, area_cm2)
      coverage_percent : % of frame classified as leaf/canopy (always returned)
      area_cm2         : absolute area only if frame_area_cm2 is calibrated
    """
    from PIL import Image

    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((256, 256))
        arr = np.array(img).astype(float) / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        maxc = np.max(arr, axis=2)
        minc = np.min(arr, axis=2)
        diff = np.where(maxc - minc == 0, 1e-6, maxc - minc)

        mask_r = (maxc == r)
        mask_g = (maxc == g) & ~mask_r
        mask_b = (maxc == b) & ~mask_r & ~mask_g

        hue = np.zeros_like(maxc)
        hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
        hue[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120
        hue[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240

        sat = np.where(maxc == 0, 0, (maxc - minc) / (maxc + 1e-6))
        val = maxc

        leaf_mask = (hue >= 35) & (hue <= 170) & (sat > 0.15) & (val > 0.10)
        coverage_percent = round(float(leaf_mask.mean() * 100), 2)
        area_cm2 = (
            round(coverage_percent / 100 * frame_area_cm2, 2)
            if frame_area_cm2 else None
        )
        return coverage_percent, area_cm2

    except Exception as e:
        print(f"[ML] Leaf area error: {e}")
        return None, None


def extract_image_features(image_path, frame_area_cm2=None):
    """
    Convenience wrapper: run both extraction functions and return a dict
    ready to be unpacked into predict_growth(**kwargs).

    Returns
    -------
    dict with keys: greenness_score, leaf_coverage_percent, leaf_area_norm,
                    leaf_area_cm2 (None if no calibration)
    """
    greenness = extract_greenness(image_path)
    coverage, area_cm2 = extract_leaf_area(image_path, frame_area_cm2)

    # Fallback if leaf extraction fails
    if coverage is None:
        coverage = _IMG_DEFAULTS['leaf_coverage_percent']

    leaf_area_norm = round(coverage / 100.0, 4)

    return {
        'greenness_score':       greenness,
        'leaf_coverage_percent': coverage,
        'leaf_area_norm':        leaf_area_norm,
        'leaf_area_cm2':         area_cm2,
    }


# ─── DISEASE DETECTION ─────────────────────────────────────────────────────────

_DEFAULT_DISEASE_CLASSES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]

DISEASE_MODEL_PATH = MODEL_DIR / 'disease_model.pth'
CLASSES_FILE       = MODEL_DIR / 'disease_classes.txt'
ACCURACY_FILE      = MODEL_DIR / 'disease_model_accuracy.txt'
_CNN_MODEL_ACCURACY = None


def get_disease_classes():
    if CLASSES_FILE.exists():
        classes = [c.strip() for c in CLASSES_FILE.read_text().splitlines() if c.strip()]
        if classes:
            return classes
    return _DEFAULT_DISEASE_CLASSES


def get_disease_model_accuracy():
    global _CNN_MODEL_ACCURACY
    if _CNN_MODEL_ACCURACY is not None:
        return _CNN_MODEL_ACCURACY
    if ACCURACY_FILE.exists():
        try:
            _CNN_MODEL_ACCURACY = float(ACCURACY_FILE.read_text().strip())
            return _CNN_MODEL_ACCURACY
        except Exception:
            pass
    return None


# ── Treatment databases ────────────────────────────────────────────────────────

RICE_TREATMENT = {
    "BrownSpot": {
        "pesticide": "Mancozeb 75% WP or Propiconazole 25% EC",
        "dose": "25 g Mancozeb per 10 litres  OR  1 ml Propiconazole per litre",
        "per_acre": "2.5 kg Mancozeb/acre  OR  200 ml Propiconazole/acre",
        "spray_interval": "Every 14 days, 2–3 sprays",
        "additional": "Apply potash (MOP) 20 kg/acre to strengthen immunity. Remove infected leaves."
    },
    "Hispa": {
        "pesticide": "Chlorpyrifos 20% EC or Cypermethrin 10% EC",
        "dose": "2 ml Chlorpyrifos per litre  OR  1 ml Cypermethrin per litre",
        "per_acre": "400 ml Chlorpyrifos/acre  OR  200 ml Cypermethrin/acre",
        "spray_interval": "2 sprays at 10-day interval when infestation is seen",
        "additional": "Clip and destroy affected leaf tips. Avoid dense planting."
    },
    "LeafBlast": {
        "pesticide": "Tricyclazole 75% WP (most effective for blast)",
        "dose": "6 g per 10 litres water",
        "per_acre": "600 g/acre",
        "spray_interval": "2 sprays — at tillering stage and panicle initiation",
        "additional": "Avoid excess nitrogen fertilizer. Ensure proper field drainage."
    },
    "Healthy": {
        "pesticide": "No pesticide needed",
        "dose": "—", "per_acre": "—",
        "spray_interval": "Preventive spray optional after 30 days",
        "additional": "Plant is healthy. Continue regular monitoring twice daily."
    },
}

PLANTVILLAGE_TREATMENT = {
    "Apple___Apple_scab": {"pesticide":"Captan 50% WP or Myclobutanil 10% WP","dose":"2 g Captan per litre","per_acre":"800 g Captan/acre","spray_interval":"10–14 day intervals from bud-break","additional":"Rake and destroy fallen leaves. Improve canopy airflow."},
    "Apple___Black_rot": {"pesticide":"Captan 50% WP or Thiophanate-methyl 70% WP","dose":"2 g Captan per litre","per_acre":"800 g/acre","spray_interval":"Every 10–14 days during fruit development","additional":"Prune out cankered wood and mummified fruit."},
    "Apple___Cedar_apple_rust": {"pesticide":"Myclobutanil 10% WP or Propiconazole 25% EC","dose":"1 g Myclobutanil per litre","per_acre":"400 g/acre","spray_interval":"Start at pink-bud stage, repeat every 10 days for 3 sprays","additional":"Remove nearby juniper/cedar hosts if possible."},
    "Apple___healthy": {"pesticide":"No pesticide needed","dose":"—","per_acre":"—","spray_interval":"Routine preventive copper spray after heavy rain (optional)","additional":"Tree is healthy."},
    "Blueberry___healthy": {"pesticide":"No pesticide needed","dose":"—","per_acre":"—","spray_interval":"—","additional":"Plant is healthy."},
    "Cherry_(including_sour)___Powdery_mildew": {"pesticide":"Sulfur 80% WP or Myclobutanil 10% WP","dose":"2 g Sulfur per litre","per_acre":"800 g/acre","spray_interval":"Every 7–10 days once symptoms appear","additional":"Avoid excess nitrogen. Improve air circulation via pruning."},
    "Cherry_(including_sour)___healthy": {"pesticide":"No pesticide needed","dose":"—","per_acre":"—","spray_interval":"—","additional":"Tree is healthy."},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"pesticide":"Azoxystrobin 23% SC or Propiconazole 25% EC","dose":"1 ml per litre","per_acre":"400 ml/acre","spray_interval":"At first symptoms and again 14 days later","additional":"Rotate away from maize for 1 season."},
    "Corn_(maize)___Common_rust_": {"pesticide":"Mancozeb 75% WP or Propiconazole 25% EC","dose":"2.5 g Mancozeb per litre","per_acre":"1 kg/acre","spray_interval":"Every 10–14 days if humid weather continues","additional":"Use rust-resistant hybrids next season."},
    "Corn_(maize)___Northern_Leaf_Blight": {"pesticide":"Propiconazole 25% EC or Azoxystrobin 23% SC","dose":"1 ml per litre","per_acre":"400 ml/acre","spray_interval":"At tasseling if lesions are spreading, repeat after 14 days","additional":"Rotate crops and plough under residue."},
    "Corn_(maize)___healthy": {"pesticide":"No pesticide needed","dose":"—","per_acre":"—","spray_interval":"—","additional":"Crop is healthy."},
    "Grape___Black_rot": {"pesticide":"Mancozeb 75% WP or Myclobutanil 10% WP","dose":"2.5 g Mancozeb per litre","per_acre":"1 kg/acre","spray_interval":"Every 10–14 days from bud-break to veraison","additional":"Remove mummified berries during winter pruning."},
    "Grape___Esca_(Black_Measles)": {"pesticide":"No fully effective chemical control; Thiophanate-methyl paste on pruning wounds","dose":"As per label","per_acre":"—","spray_interval":"Apply to pruning wounds immediately","additional":"Remove and burn severely infected vines."},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"pesticide":"Mancozeb 75% WP or Copper Oxychloride 50% WP","dose":"2.5 g Mancozeb per litre","per_acre":"1 kg/acre","spray_interval":"Every 10–14 days in humid conditions","additional":"Improve canopy ventilation."},
    "Grape___healthy": {"pesticide":"No pesticide needed","dose":"—","per_acre":"—","spray_interval":"—","additional":"Vine is healthy."},
    "Orange___Haunglongbing_(Citrus_greening)": {"pesticide":"No cure — manage Asian citrus psyllid with Imidacloprid 17.8% SL","dose":"0.5 ml per litre","per_acre":"As per label","spray_interval":"Every 3–4 weeks during psyllid activity","additional":"Remove and destroy infected trees; no chemical cure exists."},
    "Peach___Bacterial_spot": {"pesticide":"Copper Oxychloride 50% WP","dose":"3 g per litre","per_acre":"1.2 kg/acre","spray_interval":"Every 10–14 days during wet periods","additional":"Avoid overhead irrigation. Prune for airflow."},
    "Peach___healthy": {"pesticide":"No pesticide needed","dose":"—","per_acre":"—","spray_interval":"—","additional":"Tree is healthy."},
    "Pepper,_bell___Bacterial_spot": {"pesticide":"Copper Oxychloride 50% WP + Mancozeb","dose":"3 g Copper + 2 g Mancozeb per litre","per_acre":"1.2 kg Copper + 800 g Mancozeb/acre","spray_interval":"Every 7–10 days in wet weather","additional":"Use disease-free seed and drip irrigation."},
    "Pepper,_bell___healthy": {"pesticide":"No pesticide needed","dose":"—","per_acre":"—","spray_interval":"—","additional":"Plant is healthy."},
    "Potato___Early_blight": {"pesticide":"Mancozeb 75% WP or Chlorothalonil 75% WP","dose":"2.5 g Mancozeb per litre","per_acre":"1 kg/acre","spray_interval":"Every 7–10 days once lower-leaf spotting starts","additional":"Rotate out of potato/tomato for 2 years."},
    "Potato___Late_blight": {"pesticide":"Metalaxyl + Mancozeb (Ridomil-type) or Cymoxanil + Mancozeb","dose":"2.5 g per litre","per_acre":"1 kg/acre","spray_interval":"Every 5–7 days in cool, wet weather","additional":"Destroy infected haulms/tubers immediately."},
    "Potato___healthy": {"pesticide":"No pesticide needed","dose":"—","per_acre":"—","spray_interval":"—","additional":"Plant is healthy."},
    "Raspberry___healthy": {"pesticide":"No pesticide needed","dose":"—","per_acre":"—","spray_interval":"—","additional":"Plant is healthy."},
    "Soybean___healthy": {"pesticide":"No pesticide needed","dose":"—","per_acre":"—","spray_interval":"—","additional":"Plant is healthy."},
    "Squash___Powdery_mildew": {"pesticide":"Sulfur 80% WP or Potassium Bicarbonate spray","dose":"2 g Sulfur per litre","per_acre":"800 g/acre","spray_interval":"Every 7 days once patches appear","additional":"Avoid overhead watering late in the day."},
    "Strawberry___Leaf_scorch": {"pesticide":"Captan 50% WP or Myclobutanil 10% WP","dose":"2 g Captan per litre","per_acre":"800 g/acre","spray_interval":"Every 10–14 days during active growth","additional":"Remove old/infected leaves after harvest. Use drip irrigation."},
    "Strawberry___healthy": {"pesticide":"No pesticide needed","dose":"—","per_acre":"—","spray_interval":"—","additional":"Plant is healthy."},
    "Tomato___Bacterial_spot": {"pesticide":"Copper Oxychloride 50% WP + Mancozeb","dose":"3 g Copper + 2 g Mancozeb per litre","per_acre":"1.2 kg Copper + 800 g Mancozeb/acre","spray_interval":"Every 7–10 days","additional":"Use disease-free seed, avoid overhead irrigation."},
    "Tomato___Early_blight": {"pesticide":"Mancozeb 75% WP or Chlorothalonil 75% WP","dose":"2.5 g Mancozeb per litre","per_acre":"1 kg/acre","spray_interval":"Every 7–10 days once lower-leaf spotting starts","additional":"Remove lower infected leaves, stake plants for airflow."},
    "Tomato___Late_blight": {"pesticide":"Metalaxyl + Mancozeb combi or Cymoxanil + Mancozeb","dose":"2.5 g per litre","per_acre":"1 kg/acre","spray_interval":"Every 5–7 days in cool, wet weather","additional":"Remove and destroy infected plants immediately."},
    "Tomato___Leaf_Mold": {"pesticide":"Chlorothalonil 75% WP or Copper Oxychloride 50% WP","dose":"2 g per litre","per_acre":"800 g/acre","spray_interval":"Every 7–10 days","additional":"Reduce humidity, improve ventilation."},
    "Tomato___Septoria_leaf_spot": {"pesticide":"Mancozeb 75% WP or Chlorothalonil 75% WP","dose":"2.5 g per litre","per_acre":"1 kg/acre","spray_interval":"Every 7–10 days","additional":"Remove infected lower leaves, avoid overhead watering."},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"pesticide":"Abamectin 1.8% EC or Spiromesifen 22.9% SC","dose":"0.5 ml per litre","per_acre":"200 ml/acre","spray_interval":"2 sprays 7 days apart; spray undersides of leaves","additional":"Increase humidity; avoid broad-spectrum insecticides."},
    "Tomato___Target_Spot": {"pesticide":"Azoxystrobin 23% SC or Chlorothalonil 75% WP","dose":"1 ml Azoxystrobin per litre","per_acre":"400 ml/acre","spray_interval":"Every 7–10 days","additional":"Improve airflow, remove infected leaf debris."},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"pesticide":"No cure — control whitefly with Imidacloprid 17.8% SL or Thiamethoxam 25% WG","dose":"0.3 g Thiamethoxam per litre","per_acre":"120 g/acre","spray_interval":"Every 10 days during whitefly activity","additional":"Remove and destroy infected plants; use resistant varieties."},
    "Tomato___Tomato_mosaic_virus": {"pesticide":"No chemical cure — sanitation and resistant varieties only","dose":"—","per_acre":"—","spray_interval":"—","additional":"Remove infected plants. Disinfect tools. Use resistant (Tm) varieties."},
    "Tomato___healthy": {"pesticide":"No pesticide needed","dose":"—","per_acre":"—","spray_interval":"—","additional":"Plant is healthy."},
}

GENERIC_UNMAPPED_TREATMENT = {
    "pesticide": "Treatment not catalogued for this predicted class yet",
    "dose": "—", "per_acre": "—", "spray_interval": "—",
    "additional": (
        "The predicted class has no treatment entry in the database. "
        "Consult your local agricultural extension officer."
    ),
}


def predict_disease(image_path):
    """
    Run ResNet-18 disease classifier.

    Returns
    -------
    (diseased, disease_name, confidence, model_acc, treatment_dict, low_confidence)
    """
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from PIL import Image

    classes    = get_disease_classes()
    model_acc  = get_disease_model_accuracy()
    n_classes  = len(classes)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Build model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, n_classes),
    )

    if not DISEASE_MODEL_PATH.exists():
        return False, "Model not found", 0.0, None, GENERIC_UNMAPPED_TREATMENT, True

    device = torch.device('cpu')
    state  = torch.load(DISEASE_MODEL_PATH, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()

    img   = Image.open(image_path).convert('RGB')
    x     = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze()

    conf_val, pred_idx = probs.max(0)
    confidence    = round(float(conf_val) * 100, 1)
    disease_name  = classes[int(pred_idx)]
    low_confidence = confidence < 60

    # Lookup treatment
    is_healthy = 'healthy' in disease_name.lower()
    diseased   = not is_healthy

    treatment = (
        RICE_TREATMENT.get(disease_name)
        or PLANTVILLAGE_TREATMENT.get(disease_name)
        or GENERIC_UNMAPPED_TREATMENT
    )

    # Warn if the model was never trained on Green Gram
    if not any(c.lower() in ('brownspot', 'hispa', 'leafblast', 'healthy') for c in classes):
        treatment = dict(treatment)
        treatment['additional'] = (
            "[Note: model trained on non-Green Gram classes — treat result as "
            "closest visual match only] " + treatment.get('additional', '')
        )
        low_confidence = True

    return diseased, disease_name, confidence, model_acc, treatment, low_confidence
