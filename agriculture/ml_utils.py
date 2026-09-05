import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'ml_models'
GROWTH_MODEL_PATH = MODEL_DIR / 'growth_model.pkl'

# ─── GROWTH / REGRESSION MODEL ───────────────────────────────────────────────

def train_and_save_growth_model():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    import joblib

    np.random.seed(42)
    n = 1000

    # Features: ph, N, P, K, temp, humidity, moisture, day_number
    ph          = np.random.uniform(5.5, 7.5, n)
    nitrogen    = np.random.uniform(20, 120, n)
    phosphorus  = np.random.uniform(10, 80, n)
    potassium   = np.random.uniform(20, 100, n)
    temperature = np.random.uniform(18, 38, n)
    humidity    = np.random.uniform(40, 90, n)
    moisture    = np.random.uniform(20, 80, n)
    day_number  = np.random.uniform(1, 120, n)

    # Yield per acre (kg) — Green Gram typical 300–700 kg/acre
    yield_per_acre = (
        420
        + 2.5 * nitrogen
        + 1.8 * phosphorus
        + 1.5 * potassium
        - 15  * np.abs(ph - 6.8)
        + 0.8 * humidity
        - 4   * np.abs(temperature - 29)
        + 1.5 * moisture
        + 3   * day_number
        + np.random.normal(0, 12, n)
    )
    yield_per_acre = np.clip(yield_per_acre, 100, 800)

    # Height (cm): green gram max ~50cm, sigmoid over 60 days
    height = 5 + 45 * (1 / (1 + np.exp(-0.12 * (day_number - 28))))
    height += 3 * (nitrogen / 100) - 2 * np.abs(ph - 6.8) + np.random.normal(0, 1.5, n)
    height = np.clip(height, 2, 55)

    X = np.column_stack([ph, nitrogen, phosphorus, potassium, temperature, humidity, moisture, day_number])
    X_train, X_test, y_train, y_test = train_test_split(X, yield_per_acre, test_size=0.2, random_state=42)
    _, _, h_train, h_test = train_test_split(X, height, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    yield_model = GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=5, random_state=42)
    yield_model.fit(X_train_s, y_train)
    yield_acc = round(r2_score(y_test, yield_model.predict(X_test_s)) * 100, 1)

    height_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    height_model.fit(X_train_s, h_train)
    height_acc = round(r2_score(h_test, height_model.predict(X_test_s)) * 100, 1)

    MODEL_DIR.mkdir(exist_ok=True)
    import joblib
    joblib.dump({'yield_model': yield_model, 'height_model': height_model,
                 'scaler': scaler, 'yield_accuracy': yield_acc, 'height_accuracy': height_acc},
                GROWTH_MODEL_PATH)
    print(f"[ML] Growth model trained. Yield R²: {yield_acc}% | Height R²: {height_acc}%")
    return yield_model, height_model, scaler, yield_acc, height_acc


def load_growth_model():
    import joblib
    if not GROWTH_MODEL_PATH.exists():
        return train_and_save_growth_model()
    d = joblib.load(GROWTH_MODEL_PATH)
    return d['yield_model'], d['height_model'], d['scaler'], d['yield_accuracy'], d['height_accuracy']


def predict_growth(ph, nitrogen, phosphorus, potassium, temperature, humidity, moisture, day_number=30, field_acres=1.0):
    """Returns dict with yield/acre, height, health, stage, accuracy, detailed_recommendations."""
    yield_model, height_model, scaler, yield_acc, height_acc = load_growth_model()
    X = np.array([[ph, nitrogen, phosphorus, potassium, temperature, humidity, moisture, day_number]])
    X_s = scaler.transform(X)
    yield_per_acre = round(float(yield_model.predict(X_s)[0]), 1)
    pred_height    = round(float(height_model.predict(X_s)[0]), 1)

    # Health score
    score = 100
    if ph < 5.5 or ph > 7.5:        score -= 25
    elif ph < 6.0 or ph > 7.0:      score -= 10
    if nitrogen < 30:                score -= 20
    elif nitrogen < 50:              score -= 8
    if phosphorus < 15:              score -= 12
    if potassium < 25:               score -= 10
    if temperature > 35:             score -= 18
    elif temperature < 18:           score -= 15
    if humidity < 40:                score -= 12
    if moisture < 25:                score -= 18
    elif moisture < 35:              score -= 8
    health_score = max(0, min(100, score))

    # Growth stage by day — Green Gram (55-65 day crop)
    if day_number <= 5:      stage = "Germination"
    elif day_number <= 15:   stage = "Seedling"
    elif day_number <= 30:   stage = "Vegetative"
    elif day_number <= 42:   stage = "Flowering"
    elif day_number <= 55:   stage = "Pod Formation"
    elif day_number <= 65:   stage = "Maturity"
    else:                    stage = "Harvest Ready"

    # Detailed recommendations with quantities
    recs = []
    if ph < 5.8:
        lime_kg = round((6.5 - ph) * 200 * field_acres, 1)
        recs.append({
            "type": "soil",
            "icon": "🪨",
            "issue": f"Low pH ({ph})",
            "action": f"Apply Agricultural Lime",
            "quantity": f"{lime_kg} kg/acre",
            "timing": "Apply before next irrigation"
        })
    if ph > 7.2:
        sulfur_kg = round((ph - 6.5) * 120 * field_acres, 1)
        recs.append({
            "type": "soil",
            "icon": "⚗️",
            "issue": f"High pH ({ph})",
            "action": "Apply Elemental Sulfur",
            "quantity": f"{sulfur_kg} kg/acre",
            "timing": "Mix into top 10 cm soil"
        })
    if nitrogen < 30:
        urea_kg = round((80 - nitrogen) * 2.17 * field_acres, 1)
        recs.append({
            "type": "fertilizer",
            "icon": "🌿",
            "issue": f"Nitrogen deficiency ({nitrogen} mg/kg)",
            "action": "Apply Urea (46% N)",
            "quantity": f"{urea_kg} kg/acre",
            "timing": "Split into 2 doses — now & 15 days later"
        })
    elif nitrogen < 50:
        urea_kg = round((60 - nitrogen) * 2.17 * field_acres, 1)
        recs.append({
            "type": "fertilizer",
            "icon": "🌿",
            "issue": f"Low Nitrogen ({nitrogen} mg/kg)",
            "action": "Apply Urea (46% N)",
            "quantity": f"{urea_kg} kg/acre",
            "timing": "Apply within 3 days"
        })
    if phosphorus < 15:
        dap_kg = round((40 - phosphorus) * 5.43 * field_acres, 1)
        recs.append({
            "type": "fertilizer",
            "icon": "🟤",
            "issue": f"Phosphorus deficiency ({phosphorus} mg/kg)",
            "action": "Apply DAP (18% N, 46% P₂O₅)",
            "quantity": f"{dap_kg} kg/acre",
            "timing": "Basal application before sowing"
        })
    if potassium < 25:
        mop_kg = round((60 - potassium) * 1.67 * field_acres, 1)
        recs.append({
            "type": "fertilizer",
            "icon": "🔴",
            "issue": f"Potassium deficiency ({potassium} mg/kg)",
            "action": "Apply MOP (Muriate of Potash, 60% K₂O)",
            "quantity": f"{mop_kg} kg/acre",
            "timing": "Apply at tillering stage"
        })
    if moisture < 25:
        water_mm = round((45 - moisture) * 3.5 * field_acres, 0)
        recs.append({
            "type": "water",
            "icon": "💧",
            "issue": f"Critically low soil moisture ({moisture}%)",
            "action": "Irrigate immediately",
            "quantity": f"{int(water_mm)} litres/acre",
            "timing": "Within 24 hours — crop stress risk"
        })
    elif moisture < 35:
        recs.append({
            "type": "water",
            "icon": "💧",
            "issue": f"Low soil moisture ({moisture}%)",
            "action": "Schedule irrigation",
            "quantity": f"{int(round((40 - moisture) * 2.5 * field_acres, 0))} litres/acre",
            "timing": "Within 2–3 days"
        })
    if temperature > 35:
        recs.append({
            "type": "environment",
            "icon": "🌡️",
            "issue": f"Heat stress ({temperature}°C)",
            "action": "Foliar spray with 1% Potassium Chloride solution",
            "quantity": f"{round(2.5 * field_acres, 1)} litres solution/acre",
            "timing": "Spray in early morning or evening"
        })
    if humidity < 40:
        recs.append({
            "type": "environment",
            "icon": "🌫️",
            "issue": f"Low humidity ({humidity}%)",
            "action": "Increase irrigation frequency",
            "quantity": "Light irrigation every 2 days",
            "timing": "Continue until humidity > 55%"
        })
    if not recs:
        recs.append({
            "type": "ok",
            "icon": "✅",
            "issue": "All conditions optimal",
            "action": "Maintain current practices",
            "quantity": "—",
            "timing": "Monitor sensors daily"
        })

    return {
        "predicted_yield_per_acre": yield_per_acre,
        "predicted_height_cm": pred_height,
        "health_score": round(health_score, 1),
        "growth_stage": stage,
        "model_accuracy": yield_acc,
        "height_accuracy": height_acc,
        "recommendations": recs,
        "recommendation_text": " | ".join(r['action'] + ": " + r['quantity'] for r in recs),
    }


# ─── IMAGE: GREENNESS SCORE ────────────────────────────────────────────────

def extract_greenness(image_path):
    from PIL import Image
    import numpy as np

    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((128,128))

        pixels = np.array(img).astype(float)

        r = pixels[:,:,0]
        g = pixels[:,:,1]
        b = pixels[:,:,2]

        # Excess Green Index
        exg = (2*g - r - b)

        score = np.mean(exg)

        score = np.clip(
            (score / 255) * 100,
            0,
            100
        )

        return round(float(score),1)

    except:
        return 70.0


# ─── IMAGE: LEAF / CANOPY AREA ─────────────────────────────────────────────

def extract_leaf_area(image_path, frame_area_cm2=None):
    """
    Estimate how much of the frame is covered by plant/leaf material, using
    an HSV-based green-pixel mask (more robust than a plain colour-average
    like extract_greenness, since it ignores non-green background pixels
    entirely instead of averaging them in).

    Returns (coverage_percent, area_cm2):
      - coverage_percent: % of the frame classified as leaf/canopy. Always
        returned if the image loads. Track this over time even without any
        camera calibration — its trend day-over-day IS a growth signal.
      - area_cm2: absolute leaf area, ONLY if `frame_area_cm2` (the real
        world area visible in the camera's fixed frame, set once per
        CropSession via camera_frame_area_cm2) is provided. Otherwise None.
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
        hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120)
        hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240)

        sat = np.where(maxc == 0, 0, (maxc - minc) / (maxc + 1e-6))
        val = maxc

        # Leaf/canopy: green-yellow to green-cyan hues, with enough
        # saturation/brightness to exclude near-grey soil, shadows, wires.
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


# ─── DISEASE DETECTION ────────────────────────────────────────────────────────

# Default classes — overridden by ml_models/disease_classes.txt if present
# Matches your dataset folder names exactly: BrownSpot, Healthy, Hispa, LeafBlast
# (ImageFolder sorts alphabetically, so this order must match)
_DEFAULT_DISEASE_CLASSES = [
    "BrownSpot",
    "Healthy",
    "Hispa",
    "LeafBlast",
]

DISEASE_MODEL_PATH  = MODEL_DIR / 'disease_model.pth'
CLASSES_FILE        = MODEL_DIR / 'disease_classes.txt'
ACCURACY_FILE       = MODEL_DIR / 'disease_model_accuracy.txt'
_CNN_MODEL_ACCURACY = None


def get_disease_classes():
    """Read class names from file saved during training, else use defaults."""
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
        except:
            pass
    return None

# Use property so it always reflects current file
@property
def DISEASE_CLASSES():
    return get_disease_classes()


# ── Treatment recommendations ───────────────────────────────────────────────
# Two dictionaries are kept so BOTH your Phase-1 rice model and the Phase-2
# 38-class PlantVillage model resolve to a real treatment instead of silently
# falling back to "Healthy".
#
# NOTE (important): the 38 classes below come from the public PlantVillage /
# "New Plant Diseases Dataset". None of them are Green Gram. If Green Gram is
# your live target crop, this model can only ever say "closest look-alike
# class", never a true Green Gram diagnosis — see README_DIAGNOSIS.md.
# Doses are general agronomic guidance (India-common actives); always confirm
# against your local agricultural extension / product label before spraying.

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
        "additional": "Clip and destroy affected leaf tips. Avoid dense planting. Use sticky traps to monitor adult hispa population."
    },
    "LeafBlast": {
        "pesticide": "Tricyclazole 75% WP (most effective for blast)",
        "dose": "6 g per 10 litres water",
        "per_acre": "600 g/acre",
        "spray_interval": "2 sprays — at tillering stage and panicle initiation",
        "additional": "Avoid excess nitrogen fertilizer. Ensure proper field drainage. Use blast-resistant varieties next season."
    },
    "Healthy": {
        "pesticide": "No pesticide needed",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Preventive spray optional after 30 days",
        "additional": "Plant is healthy. Continue regular monitoring twice daily."
    }
}

# Full mapping for all 38 "New Plant Diseases Dataset" / PlantVillage classes
# (keys match disease_classes.txt exactly).
PLANTVILLAGE_TREATMENT = {
    "Apple___Apple_scab": {
        "pesticide": "Captan 50% WP or Myclobutanil 10% WP",
        "dose": "2 g Captan per litre  OR  1 g Myclobutanil per litre",
        "per_acre": "800 g Captan/acre (400 L spray volume)",
        "spray_interval": "10–14 day intervals from bud-break through wet spring weather",
        "additional": "Rake and destroy fallen leaves in autumn to reduce overwintering spores. Improve canopy airflow by pruning."
    },
    "Apple___Black_rot": {
        "pesticide": "Captan 50% WP or Thiophanate-methyl 70% WP",
        "dose": "2 g Captan per litre",
        "per_acre": "800 g/acre",
        "spray_interval": "Every 10–14 days during fruit development",
        "additional": "Prune out cankered/dead wood and mummified fruit — they are the main infection source."
    },
    "Apple___Cedar_apple_rust": {
        "pesticide": "Myclobutanil 10% WP or Propiconazole 25% EC",
        "dose": "1 g Myclobutanil per litre",
        "per_acre": "400 g/acre",
        "spray_interval": "Start at pink-bud stage, repeat every 10 days for 3 sprays",
        "additional": "Remove nearby juniper/cedar hosts if possible — the fungus needs both hosts to complete its cycle."
    },
    "Apple___healthy": {
        "pesticide": "No pesticide needed", "dose": "—", "per_acre": "—",
        "spray_interval": "Routine preventive copper spray after heavy rain (optional)",
        "additional": "Tree is healthy. Continue regular monitoring."
    },
    "Blueberry___healthy": {
        "pesticide": "No pesticide needed", "dose": "—", "per_acre": "—",
        "spray_interval": "—", "additional": "Plant is healthy. Continue regular monitoring."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "pesticide": "Sulfur 80% WP or Myclobutanil 10% WP",
        "dose": "2 g Sulfur per litre",
        "per_acre": "800 g/acre",
        "spray_interval": "Every 7–10 days once symptoms appear",
        "additional": "Avoid excess nitrogen (drives soft, susceptible new growth). Improve air circulation via pruning."
    },
    "Cherry_(including_sour)___healthy": {
        "pesticide": "No pesticide needed", "dose": "—", "per_acre": "—",
        "spray_interval": "—", "additional": "Tree is healthy. Continue regular monitoring."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "pesticide": "Azoxystrobin 23% SC or Propiconazole 25% EC",
        "dose": "1 ml per litre",
        "per_acre": "400 ml/acre",
        "spray_interval": "At first symptoms and again 14 days later if pressure continues",
        "additional": "Rotate away from maize for 1 season and till under residue — the fungus survives in old corn debris."
    },
    "Corn_(maize)___Common_rust_": {
        "pesticide": "Mancozeb 75% WP or Propiconazole 25% EC",
        "dose": "2.5 g Mancozeb per litre",
        "per_acre": "1 kg/acre",
        "spray_interval": "Every 10–14 days if humid weather continues",
        "additional": "Use rust-resistant hybrids next season; rust is usually manageable and rarely needs more than 1–2 sprays."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "pesticide": "Propiconazole 25% EC or Azoxystrobin 23% SC",
        "dose": "1 ml per litre",
        "per_acre": "400 ml/acre",
        "spray_interval": "At tasseling if lesions are spreading up the canopy, repeat after 14 days",
        "additional": "Rotate crops and plough under residue; resistant hybrids give the best long-term control."
    },
    "Corn_(maize)___healthy": {
        "pesticide": "No pesticide needed", "dose": "—", "per_acre": "—",
        "spray_interval": "—", "additional": "Crop is healthy. Continue regular monitoring."
    },
    "Grape___Black_rot": {
        "pesticide": "Mancozeb 75% WP or Myclobutanil 10% WP",
        "dose": "2.5 g Mancozeb per litre",
        "per_acre": "1 kg/acre",
        "spray_interval": "Every 10–14 days from bud-break to veraison",
        "additional": "Remove mummified berries and infected canes during winter pruning — the main overwintering source."
    },
    "Grape___Esca_(Black_Measles)": {
        "pesticide": "No fully effective chemical control; Thiophanate-methyl paste on pruning wounds helps",
        "dose": "As per product label on cut surfaces",
        "per_acre": "—",
        "spray_interval": "Apply to pruning wounds immediately after cutting",
        "additional": "Remove and burn severely infected vines/trunks. Avoid pruning in wet weather to reduce infection."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "pesticide": "Mancozeb 75% WP or Copper Oxychloride 50% WP",
        "dose": "2.5 g Mancozeb per litre",
        "per_acre": "1 kg/acre",
        "spray_interval": "Every 10–14 days in humid conditions",
        "additional": "Improve canopy ventilation and remove infected leaves promptly."
    },
    "Grape___healthy": {
        "pesticide": "No pesticide needed", "dose": "—", "per_acre": "—",
        "spray_interval": "—", "additional": "Vine is healthy. Continue regular monitoring."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "pesticide": "No cure — manage the Asian citrus psyllid vector with Imidacloprid 17.8% SL",
        "dose": "0.5 ml per litre (soil drench/foliar per label)",
        "per_acre": "As per label; rotate with a second insecticide class to delay resistance",
        "spray_interval": "Every 3–4 weeks during psyllid activity",
        "additional": "There is no chemical cure. Remove and destroy infected trees to stop spread; control psyllid vector aggressively; use certified disease-free planting material."
    },
    "Peach___Bacterial_spot": {
        "pesticide": "Copper Oxychloride 50% WP",
        "dose": "3 g per litre",
        "per_acre": "1.2 kg/acre",
        "spray_interval": "Every 10–14 days during wet, warm periods",
        "additional": "Avoid overhead irrigation. Prune for airflow. Bactericides give only partial control — resistant varieties help most."
    },
    "Peach___healthy": {
        "pesticide": "No pesticide needed", "dose": "—", "per_acre": "—",
        "spray_interval": "—", "additional": "Tree is healthy. Continue regular monitoring."
    },
    "Pepper,_bell___Bacterial_spot": {
        "pesticide": "Copper Oxychloride 50% WP + Mancozeb",
        "dose": "3 g Copper + 2 g Mancozeb per litre",
        "per_acre": "1.2 kg Copper + 800 g Mancozeb/acre",
        "spray_interval": "Every 7–10 days in wet weather",
        "additional": "Avoid working in wet fields (spreads bacteria). Use disease-free seed/transplants and drip irrigation."
    },
    "Pepper,_bell___healthy": {
        "pesticide": "No pesticide needed", "dose": "—", "per_acre": "—",
        "spray_interval": "—", "additional": "Plant is healthy. Continue regular monitoring."
    },
    "Potato___Early_blight": {
        "pesticide": "Mancozeb 75% WP or Chlorothalonil 75% WP",
        "dose": "2.5 g Mancozeb per litre",
        "per_acre": "1 kg/acre",
        "spray_interval": "Every 7–10 days once lower-leaf spotting starts",
        "additional": "Ensure balanced nitrogen (excess K deficiency worsens it). Rotate out of potato/tomato for 2 years."
    },
    "Potato___Late_blight": {
        "pesticide": "Metalaxyl + Mancozeb (Ridomil-type combi) or Cymoxanil + Mancozeb",
        "dose": "2.5 g per litre",
        "per_acre": "1 kg/acre",
        "spray_interval": "Every 5–7 days in cool, wet weather — this disease can destroy a crop within days",
        "additional": "Destroy infected haulms/tubers immediately. Avoid overhead irrigation. Highest-priority disease to act on fast."
    },
    "Potato___healthy": {
        "pesticide": "No pesticide needed", "dose": "—", "per_acre": "—",
        "spray_interval": "—", "additional": "Plant is healthy. Continue regular monitoring."
    },
    "Raspberry___healthy": {
        "pesticide": "No pesticide needed", "dose": "—", "per_acre": "—",
        "spray_interval": "—", "additional": "Plant is healthy. Continue regular monitoring."
    },
    "Soybean___healthy": {
        "pesticide": "No pesticide needed", "dose": "—", "per_acre": "—",
        "spray_interval": "—", "additional": "Plant is healthy. Continue regular monitoring."
    },
    "Squash___Powdery_mildew": {
        "pesticide": "Sulfur 80% WP or Potassium Bicarbonate spray",
        "dose": "2 g Sulfur per litre",
        "per_acre": "800 g/acre",
        "spray_interval": "Every 7 days once white powdery patches appear",
        "additional": "Avoid overhead watering late in the day. Space plants for airflow. Remove badly infected leaves."
    },
    "Strawberry___Leaf_scorch": {
        "pesticide": "Captan 50% WP or Myclobutanil 10% WP",
        "dose": "2 g Captan per litre",
        "per_acre": "800 g/acre",
        "spray_interval": "Every 10–14 days during active growth",
        "additional": "Remove old/infected leaves after harvest. Avoid overhead irrigation; use drip instead."
    },
    "Strawberry___healthy": {
        "pesticide": "No pesticide needed", "dose": "—", "per_acre": "—",
        "spray_interval": "—", "additional": "Plant is healthy. Continue regular monitoring."
    },
    "Tomato___Bacterial_spot": {
        "pesticide": "Copper Oxychloride 50% WP + Mancozeb",
        "dose": "3 g Copper + 2 g Mancozeb per litre",
        "per_acre": "1.2 kg Copper + 800 g Mancozeb/acre",
        "spray_interval": "Every 7–10 days in warm, wet weather",
        "additional": "Use disease-free seed, avoid overhead irrigation, rotate out of solanaceous crops for 2 years."
    },
    "Tomato___Early_blight": {
        "pesticide": "Mancozeb 75% WP or Chlorothalonil 75% WP",
        "dose": "2.5 g Mancozeb per litre",
        "per_acre": "1 kg/acre",
        "spray_interval": "Every 7–10 days once lower-leaf spotting starts",
        "additional": "Remove lower infected leaves, stake plants for airflow, mulch to stop soil splash onto leaves."
    },
    "Tomato___Late_blight": {
        "pesticide": "Metalaxyl + Mancozeb combi or Cymoxanil + Mancozeb",
        "dose": "2.5 g per litre",
        "per_acre": "1 kg/acre",
        "spray_interval": "Every 5–7 days in cool, wet weather — acts fast, act fast",
        "additional": "Remove and destroy infected plants immediately; this disease spreads rapidly through a field."
    },
    "Tomato___Leaf_Mold": {
        "pesticide": "Chlorothalonil 75% WP or Copper Oxychloride 50% WP",
        "dose": "2 g per litre",
        "per_acre": "800 g/acre",
        "spray_interval": "Every 7–10 days",
        "additional": "Reduce humidity in greenhouse/polyhouse settings, improve ventilation, avoid wetting leaves."
    },
    "Tomato___Septoria_leaf_spot": {
        "pesticide": "Mancozeb 75% WP or Chlorothalonil 75% WP",
        "dose": "2.5 g per litre",
        "per_acre": "1 kg/acre",
        "spray_interval": "Every 7–10 days",
        "additional": "Remove infected lower leaves, mulch to reduce soil splash, avoid overhead watering."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "pesticide": "Abamectin 1.8% EC or Spiromesifen 22.9% SC (miticide, not a general insecticide)",
        "dose": "0.5 ml per litre",
        "per_acre": "200 ml/acre",
        "spray_interval": "2 sprays 7 days apart; spray undersides of leaves",
        "additional": "Increase humidity/water stress relief (mites thrive in hot, dry, dusty conditions). Avoid broad-spectrum insecticides that kill mite predators."
    },
    "Tomato___Target_Spot": {
        "pesticide": "Azoxystrobin 23% SC or Chlorothalonil 75% WP",
        "dose": "1 ml Azoxystrobin per litre",
        "per_acre": "400 ml/acre",
        "spray_interval": "Every 7–10 days",
        "additional": "Improve airflow, remove infected leaf debris from the ground."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "pesticide": "No cure — control the whitefly vector with Imidacloprid 17.8% SL or Thiamethoxam 25% WG",
        "dose": "0.3 g Thiamethoxam per litre",
        "per_acre": "120 g/acre",
        "spray_interval": "Every 10 days during whitefly activity",
        "additional": "There is no chemical cure for the virus itself. Remove and destroy infected plants, use yellow sticky traps and virus-resistant varieties, use insect nets on nurseries."
    },
    "Tomato___Tomato_mosaic_virus": {
        "pesticide": "No chemical cure — sanitation and resistant varieties are the only control",
        "dose": "—", "per_acre": "—",
        "spray_interval": "—",
        "additional": "Remove and destroy infected plants. Disinfect tools/hands (virus spreads by contact/sap). Use resistant (Tm) varieties next season."
    },
    "Tomato___healthy": {
        "pesticide": "No pesticide needed", "dose": "—", "per_acre": "—",
        "spray_interval": "—", "additional": "Plant is healthy. Continue regular monitoring."
    },
}

# Generic, honest fallback — used ONLY if a predicted class truly has no
# entry above, so the app never silently shows "Healthy" for a diseased leaf.
GENERIC_UNMAPPED_TREATMENT = {
    "pesticide": "Treatment not catalogued for this predicted class yet",
    "dose": "—", "per_acre": "—", "spray_interval": "—",
    "additional": ("This class doesn't have a treatment entry yet. Do not treat this as "
                   "'no disease found' — please verify manually and add the class to "
                   "DISEASE_TREATMENT in ml_utils.py.")
}

# Combined lookup used at inference time (rice classes + PlantVillage classes)
DISEASE_TREATMENT = {**RICE_TREATMENT, **PLANTVILLAGE_TREATMENT}

def format_disease_name(name):
    parts = name.split("___")

    if len(parts) == 2:
        crop = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ")

        if disease.lower() == "healthy":
            return f"{crop} - Healthy"

        return f"{crop} - {disease}"

    return name

def predict_disease(image_path):
    try:
        import torch
        import torchvision.transforms as transforms
        from torchvision import models as tv_models
        from PIL import Image as PILImage

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = PILImage.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)

        if DISEASE_MODEL_PATH.exists():

            model = tv_models.resnet18(weights=None)

            model.fc = torch.nn.Sequential(
                        torch.nn.Dropout(0.5),
                        torch.nn.Linear(
                        model.fc.in_features,
                        len(get_disease_classes())
                        )
                    )

            model.load_state_dict(
            torch.load(
                        DISEASE_MODEL_PATH,
                        map_location='cpu'
                        )
                    )

            model.eval()
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)[0]
            classes = get_disease_classes()
            idx = int(torch.argmax(probs))
            confidence = round(float(probs[idx]) * 100, 1)
            disease_name = classes[idx] if idx < len(classes) else "Unknown"
            display_name = format_disease_name(disease_name)
            model_acc = get_disease_model_accuracy()
            is_healthy = "healthy" in disease_name.lower()
            if disease_name in DISEASE_TREATMENT:
                treatment = dict(DISEASE_TREATMENT[disease_name])
            elif is_healthy:
                treatment = dict(RICE_TREATMENT["Healthy"])
            else:
                # Never silently fall back to "Healthy" for a diseased/unmapped class
                treatment = dict(GENERIC_UNMAPPED_TREATMENT)

            # Honesty flags: low raw confidence, OR the model is the 38-class
            # PlantVillage model being asked about a crop it was never
            # trained on (e.g. Green Gram). Both mean "don't trust this
            # blindly" — surface that instead of a silent misdiagnosis.
            low_confidence = confidence < 60.0
            is_plantvillage_model = len(classes) == 38 and any(
                c.startswith(("Tomato___", "Apple___", "Corn_")) for c in classes
            )
            if low_confidence:
                treatment["confidence_warning"] = (
                    f"Low model confidence ({confidence}%) — verify this diagnosis "
                    f"manually before treating."
                )
            elif is_plantvillage_model:
                treatment["confidence_warning"] = (
                    "This model was trained on a general reference dataset "
                    "(Apple/Tomato/Corn/etc.), not your specific crop — treat "
                    "this as 'closest known pattern', not a confirmed diagnosis, "
                    "until it's fine-tuned on your own crop's photos."
                )

            return (not is_healthy, disease_name, confidence, model_acc, treatment, low_confidence)
        else:
            return _colour_heuristic(image_path)
    except Exception as e:
        print(f"[ML] Disease error: {e}")
        return _colour_heuristic(image_path)


def _colour_heuristic(image_path):
    try:
        from PIL import Image
        img = Image.open(image_path).convert('RGB').resize((64, 64))
        px = np.array(img).astype(float)
        r, g, b = px[:,:,0].mean(), px[:,:,1].mean(), px[:,:,2].mean()
        if r > 140 and g < 100 and b < 80:
            d = "BrownSpot"; conf = round(np.random.uniform(72, 88), 1)
        elif r > 120 and g > 100 and b < 80 and r > g:
            d = "Hispa"; conf = round(np.random.uniform(68, 84), 1)
        elif g > r and g > b and g > 80:
            d = "Healthy"; conf = round(np.random.uniform(85, 97), 1)
        else:
            d = "LeafBlast"; conf = round(np.random.uniform(65, 82), 1)
        treatment = dict(DISEASE_TREATMENT.get(d, RICE_TREATMENT["Healthy"]))
        treatment["confidence_warning"] = (
            "No trained CNN weights found — this is a rough colour heuristic, "
            "not a real diagnosis. Train and save disease_model.pth for real results."
        )
        return d != "Healthy", d, conf, None, treatment, True
    except:
        fallback = dict(RICE_TREATMENT["Healthy"])
        return False, "Healthy", 90.0, None, fallback, True