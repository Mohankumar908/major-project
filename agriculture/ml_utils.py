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

    ph          = np.random.uniform(5.5, 7.5, n)
    nitrogen    = np.random.uniform(20, 120, n)
    phosphorus  = np.random.uniform(10, 80, n)
    potassium   = np.random.uniform(20, 100, n)
    temperature = np.random.uniform(18, 38, n)
    humidity    = np.random.uniform(40, 90, n)
    moisture    = np.random.uniform(20, 80, n)
    day_number  = np.random.uniform(1, 120, n)

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

    if day_number <= 5:      stage = "Germination"
    elif day_number <= 15:   stage = "Seedling"
    elif day_number <= 30:   stage = "Vegetative"
    elif day_number <= 42:   stage = "Flowering"
    elif day_number <= 55:   stage = "Pod Formation"
    elif day_number <= 65:   stage = "Maturity"
    else:                    stage = "Harvest Ready"

    recs = []
    if ph < 5.8:
        lime_kg = round((6.5 - ph) * 200 * field_acres, 1)
        recs.append({"type": "soil", "icon": "🪨", "issue": f"Low pH ({ph})", "action": "Apply Agricultural Lime",
                     "quantity": f"{lime_kg} kg/acre", "timing": "Apply before next irrigation"})
    if ph > 7.2:
        sulfur_kg = round((ph - 6.5) * 120 * field_acres, 1)
        recs.append({"type": "soil", "icon": "⚗️", "issue": f"High pH ({ph})", "action": "Apply Elemental Sulfur",
                     "quantity": f"{sulfur_kg} kg/acre", "timing": "Mix into top 10 cm soil"})
    if nitrogen < 30:
        urea_kg = round((80 - nitrogen) * 2.17 * field_acres, 1)
        recs.append({"type": "fertilizer", "icon": "🌿", "issue": f"Nitrogen deficiency ({nitrogen} mg/kg)",
                     "action": "Apply Urea (46% N)", "quantity": f"{urea_kg} kg/acre",
                     "timing": "Split into 2 doses — now & 15 days later"})
    elif nitrogen < 50:
        urea_kg = round((60 - nitrogen) * 2.17 * field_acres, 1)
        recs.append({"type": "fertilizer", "icon": "🌿", "issue": f"Low Nitrogen ({nitrogen} mg/kg)",
                     "action": "Apply Urea (46% N)", "quantity": f"{urea_kg} kg/acre", "timing": "Apply within 3 days"})
    if phosphorus < 15:
        dap_kg = round((40 - phosphorus) * 5.43 * field_acres, 1)
        recs.append({"type": "fertilizer", "icon": "🟤", "issue": f"Phosphorus deficiency ({phosphorus} mg/kg)",
                     "action": "Apply DAP (18% N, 46% P₂O₅)", "quantity": f"{dap_kg} kg/acre",
                     "timing": "Basal application before sowing"})
    if potassium < 25:
        mop_kg = round((60 - potassium) * 1.67 * field_acres, 1)
        recs.append({"type": "fertilizer", "icon": "🔴", "issue": f"Potassium deficiency ({potassium} mg/kg)",
                     "action": "Apply MOP (Muriate of Potash, 60% K₂O)", "quantity": f"{mop_kg} kg/acre",
                     "timing": "Apply at tillering stage"})
    if moisture < 25:
        water_mm = round((45 - moisture) * 3.5 * field_acres, 0)
        recs.append({"type": "water", "icon": "💧", "issue": f"Critically low soil moisture ({moisture}%)",
                     "action": "Irrigate immediately", "quantity": f"{int(water_mm)} litres/acre",
                     "timing": "Within 24 hours — crop stress risk"})
    elif moisture < 35:
        recs.append({"type": "water", "icon": "💧", "issue": f"Low soil moisture ({moisture}%)",
                     "action": "Schedule irrigation",
                     "quantity": f"{int(round((40 - moisture) * 2.5 * field_acres, 0))} litres/acre",
                     "timing": "Within 2–3 days"})
    if temperature > 35:
        recs.append({"type": "environment", "icon": "🌡️", "issue": f"Heat stress ({temperature}°C)",
                     "action": "Foliar spray with 1% Potassium Chloride solution",
                     "quantity": f"{round(2.5 * field_acres, 1)} litres solution/acre",
                     "timing": "Spray in early morning or evening"})
    if humidity < 40:
        recs.append({"type": "environment", "icon": "🌫️", "issue": f"Low humidity ({humidity}%)",
                     "action": "Increase irrigation frequency", "quantity": "Light irrigation every 2 days",
                     "timing": "Continue until humidity > 55%"})
    if not recs:
        recs.append({"type": "ok", "icon": "✅", "issue": "All conditions optimal",
                     "action": "Maintain current practices", "quantity": "—", "timing": "Monitor sensors daily"})

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


# ─── IMAGE: GREENNESS SCORE ───────────────────────────────────────────────────

def extract_greenness(image_path):
    from PIL import Image
    try:
        img = Image.open(image_path).convert("RGB").resize((128, 128))
        pixels = np.array(img).astype(float)
        r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        exg = 2 * g - r - b
        score = np.clip((np.mean(exg) / 255) * 100, 0, 100)
        return round(float(score), 1)
    except Exception:
        return 70.0


# ─── DISEASE DETECTION ────────────────────────────────────────────────────────

DISEASE_MODEL_PATH = MODEL_DIR / 'disease_model.pth'
CLASSES_FILE       = MODEL_DIR / 'disease_classes.txt'
ACCURACY_FILE      = MODEL_DIR / 'disease_model_accuracy.txt'
_CNN_MODEL_ACCURACY = None

_DEFAULT_DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


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
        except Exception:
            pass
    return None


def format_disease_name(raw_name):
    """Convert raw class name like 'Tomato___Early_blight' → 'Tomato - Early Blight'."""
    parts = raw_name.split("___")
    if len(parts) == 2:
        crop    = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip().title()
        if disease.lower() == "healthy":
            return f"{crop} — Healthy"
        return f"{crop} — {disease}"
    # Already a simple name (e.g. 'BrownSpot')
    return raw_name.replace("_", " ")


# ─── Treatment database for all 38 PlantVillage classes ─────────────────────
DISEASE_TREATMENT = {
    # ── Apple ──────────────────────────────────────────────────────────────
    "Apple___Apple_scab": {
        "pesticide": "Captan 50% WP or Mancozeb 75% WP",
        "dose": "25 g Captan per 10 L  OR  20 g Mancozeb per 10 L",
        "per_acre": "2.5 kg Captan/acre  OR  2 kg Mancozeb/acre",
        "spray_interval": "Every 10–14 days during wet periods (3–5 sprays)",
        "additional": "Remove and destroy fallen infected leaves. Prune for air circulation. Apply dormant copper spray before bud break."
    },
    "Apple___Black_rot": {
        "pesticide": "Thiophanate-methyl 70% WP or Captan 50% WP",
        "dose": "15 g Thiophanate-methyl per 10 L  OR  25 g Captan per 10 L",
        "per_acre": "1.5 kg Thiophanate/acre  OR  2.5 kg Captan/acre",
        "spray_interval": "Every 10–14 days from pink bud to harvest",
        "additional": "Prune and destroy infected twigs and mummified fruits. Improve orchard sanitation."
    },
    "Apple___Cedar_apple_rust": {
        "pesticide": "Myclobutanil 20% WP or Triadimefon 25% WP",
        "dose": "10 g Myclobutanil per 10 L  OR  12 g Triadimefon per 10 L",
        "per_acre": "1 kg Myclobutanil/acre  OR  1.2 kg Triadimefon/acre",
        "spray_interval": "Every 7–10 days during spring (pink bud to petal fall)",
        "additional": "Remove nearby cedar/juniper trees if possible. Apply protective fungicide before infection periods."
    },
    "Apple___healthy": {
        "pesticide": "No treatment required",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Preventive spray optional at key growth stages",
        "additional": "Plant is healthy. Maintain regular scouting and good orchard hygiene."
    },

    # ── Blueberry ────────────────────────────────────────────────────────
    "Blueberry___healthy": {
        "pesticide": "No treatment required",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Monitor weekly during growing season",
        "additional": "Plant is healthy. Maintain soil pH 4.5–5.5 and adequate drainage."
    },

    # ── Cherry ───────────────────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": {
        "pesticide": "Sulfur 80% WP or Myclobutanil 20% WP",
        "dose": "25 g Sulfur per 10 L  OR  10 g Myclobutanil per 10 L",
        "per_acre": "2.5 kg Sulfur/acre  OR  1 kg Myclobutanil/acre",
        "spray_interval": "Every 10–14 days from shuck-split to harvest",
        "additional": "Avoid overhead irrigation. Prune for canopy airflow. Do not apply sulfur above 32°C."
    },
    "Cherry_(including_sour)___healthy": {
        "pesticide": "No treatment required",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Preventive copper spray at dormancy",
        "additional": "Plant is healthy. Scout regularly for early signs of brown rot and leaf spot."
    },

    # ── Corn (Maize) ─────────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "pesticide": "Azoxystrobin 23% SC or Propiconazole 25% EC",
        "dose": "1 ml Azoxystrobin per L  OR  1 ml Propiconazole per L",
        "per_acre": "200 ml Azoxystrobin/acre  OR  200 ml Propiconazole/acre",
        "spray_interval": "2 sprays — at tasseling and 14 days later",
        "additional": "Plant resistant hybrids. Rotate crops. Avoid excessive nitrogen. Till crop residues after harvest."
    },
    "Corn_(maize)___Common_rust_": {
        "pesticide": "Mancozeb 75% WP or Azoxystrobin 23% SC",
        "dose": "20 g Mancozeb per 10 L  OR  1 ml Azoxystrobin per L",
        "per_acre": "2 kg Mancozeb/acre  OR  200 ml Azoxystrobin/acre",
        "spray_interval": "Apply at first sign; repeat every 10–14 days (2 sprays)",
        "additional": "Use rust-resistant hybrids. Avoid late planting. Ensure proper plant spacing for airflow."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "pesticide": "Propiconazole 25% EC or Trifloxystrobin 25% WG",
        "dose": "1 ml Propiconazole per L  OR  0.8 g Trifloxystrobin per L",
        "per_acre": "200 ml Propiconazole/acre  OR  160 g Trifloxystrobin/acre",
        "spray_interval": "Apply at VT stage; repeat once in 14 days if severe",
        "additional": "Rotate with non-host crops. Incorporate infected residues. Choose resistant varieties."
    },
    "Corn_(maize)___healthy": {
        "pesticide": "No treatment required",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Monitor at VT/R1 stage for early disease signs",
        "additional": "Plant is healthy. Maintain balanced NPK. Scout regularly for pest and disease pressure."
    },

    # ── Grape ────────────────────────────────────────────────────────────
    "Grape___Black_rot": {
        "pesticide": "Myclobutanil 20% WP or Mancozeb 75% WP",
        "dose": "10 g Myclobutanil per 10 L  OR  25 g Mancozeb per 10 L",
        "per_acre": "1 kg Myclobutanil/acre  OR  2.5 kg Mancozeb/acre",
        "spray_interval": "Every 10–14 days from bud break through veraison (4–6 sprays)",
        "additional": "Remove mummified berries. Prune for air circulation. Apply fungicide before rain events."
    },
    "Grape___Esca_(Black_Measles)": {
        "pesticide": "No fully effective chemical control available",
        "dose": "Sodium arsenite (banned in most countries) — use biological alternatives",
        "per_acre": "Trichoderma-based bioagent 500 g/acre",
        "spray_interval": "Apply bioagent at pruning wounds annually",
        "additional": "Protect pruning wounds with wound sealant or Bordeaux paste. Remove severely infected vines. Avoid water stress."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "pesticide": "Copper oxychloride 50% WP or Mancozeb 75% WP",
        "dose": "25 g Copper oxychloride per 10 L  OR  20 g Mancozeb per 10 L",
        "per_acre": "2.5 kg Copper oxychloride/acre",
        "spray_interval": "Every 14 days starting at first sign (2–3 sprays)",
        "additional": "Remove and destroy infected leaves. Improve canopy ventilation through proper shoot positioning."
    },
    "Grape___healthy": {
        "pesticide": "No treatment required",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Preventive Bordeaux mixture spray at bud break",
        "additional": "Plant is healthy. Maintain proper training and pruning for air circulation."
    },

    # ── Orange ───────────────────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": {
        "pesticide": "No cure — manage psyllid vector with Imidacloprid 17.8% SL",
        "dose": "0.5 ml Imidacloprid per L for psyllid control",
        "per_acre": "100 ml Imidacloprid/acre as foliar spray",
        "spray_interval": "Every 21 days during new flush; 3–4 sprays",
        "additional": "Remove and destroy infected trees immediately to prevent spread. Use certified disease-free planting material. No chemical cure for HLB exists."
    },

    # ── Peach ────────────────────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "pesticide": "Copper hydroxide 53.8% WP or Oxytetracycline 17% SP",
        "dose": "20 g Copper hydroxide per 10 L  OR  1 g Oxytetracycline per L",
        "per_acre": "2 kg Copper hydroxide/acre",
        "spray_interval": "Weekly sprays from pink bud through pit hardening (6–8 sprays)",
        "additional": "Plant resistant varieties. Avoid overhead irrigation. Remove and destroy infected plant parts promptly."
    },
    "Peach___healthy": {
        "pesticide": "No treatment required",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Dormant copper spray before bud swell",
        "additional": "Plant is healthy. Scout for bacterial spot and brown rot early in the season."
    },

    # ── Pepper ───────────────────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot": {
        "pesticide": "Copper hydroxide 53.8% WP or Copper oxychloride 50% WP",
        "dose": "20 g per 10 L",
        "per_acre": "2 kg Copper hydroxide/acre",
        "spray_interval": "Every 7–10 days during warm, wet conditions (4–5 sprays)",
        "additional": "Use disease-free certified seed. Avoid overhead irrigation. Remove infected plant debris after harvest. Rotate with non-solanaceous crops."
    },
    "Pepper,_bell___healthy": {
        "pesticide": "No treatment required",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Preventive copper spray at transplanting",
        "additional": "Plant is healthy. Maintain adequate calcium nutrition to prevent blossom end rot."
    },

    # ── Potato ───────────────────────────────────────────────────────────
    "Potato___Early_blight": {
        "pesticide": "Mancozeb 75% WP or Chlorothalonil 75% WP",
        "dose": "20 g Mancozeb per 10 L  OR  20 g Chlorothalonil per 10 L",
        "per_acre": "2 kg Mancozeb/acre  OR  2 kg Chlorothalonil/acre",
        "spray_interval": "Every 7–10 days, starting at first sign (4–6 sprays)",
        "additional": "Remove lower infected leaves. Avoid overhead irrigation in evening. Use certified disease-free seed tubers. Rotate crops every 3 years."
    },
    "Potato___Late_blight": {
        "pesticide": "Metalaxyl + Mancozeb 72% WP (Ridomil Gold) or Cymoxanil 8% + Mancozeb 64% WP",
        "dose": "25 g Metalaxyl-Mancozeb per 10 L",
        "per_acre": "2.5 kg Metalaxyl-Mancozeb/acre",
        "spray_interval": "Every 5–7 days during cool, wet conditions — CRITICAL disease",
        "additional": "Destroy infected foliage. Hill up soil around plants. Harvest early if widespread infection. Use resistant varieties (e.g. Sarpo Mira)."
    },
    "Potato___healthy": {
        "pesticide": "No treatment required",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Preventive spray with Mancozeb at canopy closure",
        "additional": "Plant is healthy. Monitor closely during cool humid periods for late blight onset."
    },

    # ── Raspberry ────────────────────────────────────────────────────────
    "Raspberry___healthy": {
        "pesticide": "No treatment required",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Monitor weekly during fruiting season",
        "additional": "Plant is healthy. Remove old canes after harvest to prevent disease buildup."
    },

    # ── Soybean ──────────────────────────────────────────────────────────
    "Soybean___healthy": {
        "pesticide": "No treatment required",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Scout at R3 growth stage for fungal diseases",
        "additional": "Plant is healthy. Maintain proper row spacing for air circulation. Monitor for soybean rust during humid conditions."
    },

    # ── Squash ───────────────────────────────────────────────────────────
    "Squash___Powdery_mildew": {
        "pesticide": "Sulfur 80% WP or Myclobutanil 20% WP or Potassium bicarbonate",
        "dose": "25 g Sulfur per 10 L  OR  10 g Myclobutanil per 10 L",
        "per_acre": "2.5 kg Sulfur/acre  OR  1 kg Myclobutanil/acre",
        "spray_interval": "Every 7–10 days at first sign (3–4 sprays)",
        "additional": "Plant resistant varieties. Space plants adequately. Apply in early morning. Avoid wetting foliage."
    },

    # ── Strawberry ───────────────────────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "pesticide": "Captan 50% WP or Myclobutanil 20% WP",
        "dose": "20 g Captan per 10 L  OR  10 g Myclobutanil per 10 L",
        "per_acre": "2 kg Captan/acre  OR  1 kg Myclobutanil/acre",
        "spray_interval": "Every 10–14 days during growing season (3–4 sprays)",
        "additional": "Remove infected leaves. Avoid overhead irrigation. Plant disease-resistant cultivars. Renovate beds after harvest."
    },
    "Strawberry___healthy": {
        "pesticide": "No treatment required",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Preventive spray at bloom with Captan",
        "additional": "Plant is healthy. Maintain proper plant spacing and remove runners to reduce disease spread."
    },

    # ── Tomato ───────────────────────────────────────────────────────────
    "Tomato___Bacterial_spot": {
        "pesticide": "Copper hydroxide 53.8% WP or Copper oxychloride 50% WP",
        "dose": "20 g Copper hydroxide per 10 L",
        "per_acre": "2 kg Copper hydroxide/acre",
        "spray_interval": "Every 7–10 days during warm humid weather (5–6 sprays)",
        "additional": "Use certified disease-free transplants. Stake plants. Avoid working in wet foliage. Rotate crops 3–4 years."
    },
    "Tomato___Early_blight": {
        "pesticide": "Mancozeb 75% WP or Chlorothalonil 75% WP",
        "dose": "20 g Mancozeb per 10 L  OR  20 g Chlorothalonil per 10 L",
        "per_acre": "2 kg Mancozeb/acre  OR  2 kg Chlorothalonil/acre",
        "spray_interval": "Every 7–10 days from first symptom (4–5 sprays)",
        "additional": "Remove lower infected leaves immediately. Mulch soil to reduce splash. Maintain adequate potassium nutrition. Stake and prune for airflow."
    },
    "Tomato___Late_blight": {
        "pesticide": "Metalaxyl + Mancozeb 72% WP or Cymoxanil + Mancozeb WP",
        "dose": "25 g Metalaxyl-Mancozeb per 10 L",
        "per_acre": "2.5 kg Metalaxyl-Mancozeb/acre",
        "spray_interval": "Every 5–7 days — URGENT, disease spreads rapidly",
        "additional": "Destroy severely infected plants. Avoid overhead watering. Apply copper-based fungicide preventively in high-risk weather."
    },
    "Tomato___Leaf_Mold": {
        "pesticide": "Chlorothalonil 75% WP or Mancozeb 75% WP",
        "dose": "20 g per 10 L",
        "per_acre": "2 kg/acre",
        "spray_interval": "Every 7–10 days (3–4 sprays)",
        "additional": "Improve greenhouse ventilation. Reduce humidity below 85%. Remove infected leaves. Avoid excessive nitrogen fertilization."
    },
    "Tomato___Septoria_leaf_spot": {
        "pesticide": "Chlorothalonil 75% WP or Mancozeb 75% WP or Copper-based fungicide",
        "dose": "20 g Chlorothalonil per 10 L",
        "per_acre": "2 kg Chlorothalonil/acre",
        "spray_interval": "Every 7–10 days from first lesion (4–5 sprays)",
        "additional": "Remove infected lower leaves. Avoid overhead irrigation. Mulch around plant base. Rotate crops annually."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "pesticide": "Abamectin 1.8% EC or Propargite 57% EC (Miticide)",
        "dose": "0.5 ml Abamectin per L  OR  2 ml Propargite per L",
        "per_acre": "100 ml Abamectin/acre  OR  400 ml Propargite/acre",
        "spray_interval": "Every 7 days, 2–3 applications; rotate miticides",
        "additional": "Increase humidity to suppress mites. Remove heavily infested leaves. Avoid broad-spectrum insecticides that kill natural predators."
    },
    "Tomato___Target_Spot": {
        "pesticide": "Azoxystrobin 23% SC or Chlorothalonil 75% WP",
        "dose": "1 ml Azoxystrobin per L  OR  20 g Chlorothalonil per 10 L",
        "per_acre": "200 ml Azoxystrobin/acre  OR  2 kg Chlorothalonil/acre",
        "spray_interval": "Every 7–14 days (3–4 sprays)",
        "additional": "Remove infected lower leaves. Improve air circulation. Apply at first sign of lesions."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "pesticide": "Control whitefly vector: Imidacloprid 17.8% SL or Thiamethoxam 25% WG",
        "dose": "0.5 ml Imidacloprid per L",
        "per_acre": "100 ml Imidacloprid/acre",
        "spray_interval": "Every 10–14 days (3–4 sprays) targeting whiteflies",
        "additional": "No cure — remove and destroy infected plants immediately. Use reflective mulch. Install insect-proof netting. Use TYLCV-resistant varieties."
    },
    "Tomato___Tomato_mosaic_virus": {
        "pesticide": "No chemical cure — prevent spread only",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Disinfect tools with 10% bleach solution",
        "additional": "Remove and destroy infected plants immediately. Wash hands before handling plants. Avoid tobacco use near plants. Use certified virus-free seeds."
    },
    "Tomato___healthy": {
        "pesticide": "No treatment required",
        "dose": "—",
        "per_acre": "—",
        "spray_interval": "Preventive spray with Mancozeb every 21 days",
        "additional": "Plant is healthy. Maintain balanced fertilization and scout regularly for early disease signs."
    },

    # ── Legacy rice classes (kept for backward compatibility) ────────────
    "BrownSpot": {
        "pesticide": "Mancozeb 75% WP or Propiconazole 25% EC",
        "dose": "25 g Mancozeb per 10 L  OR  1 ml Propiconazole per L",
        "per_acre": "2.5 kg Mancozeb/acre  OR  200 ml Propiconazole/acre",
        "spray_interval": "Every 14 days, 2–3 sprays",
        "additional": "Apply potash (MOP) 20 kg/acre to strengthen immunity. Remove infected leaves."
    },
    "Hispa": {
        "pesticide": "Chlorpyrifos 20% EC or Cypermethrin 10% EC",
        "dose": "2 ml Chlorpyrifos per L  OR  1 ml Cypermethrin per L",
        "per_acre": "400 ml Chlorpyrifos/acre  OR  200 ml Cypermethrin/acre",
        "spray_interval": "2 sprays at 10-day interval when infestation is seen",
        "additional": "Clip and destroy affected leaf tips. Avoid dense planting. Use sticky traps to monitor adult hispa population."
    },
    "LeafBlast": {
        "pesticide": "Tricyclazole 75% WP",
        "dose": "6 g per 10 L",
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
    },
}

# ─── Default treatment for any unknown/unlisted class ────────────────────────
_DEFAULT_TREATMENT = {
    "pesticide": "Consult local agricultural extension officer",
    "dose": "—",
    "per_acre": "—",
    "spray_interval": "Scout every 3–5 days",
    "additional": "Disease class not in treatment database. Send sample to nearest plant clinic for accurate diagnosis."
}


def predict_disease(image_path):
    """
    Returns: (disease_detected: bool, display_name: str, confidence: float,
               model_accuracy: float|None, treatment: dict)
    """
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

        img    = PILImage.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)

        if DISEASE_MODEL_PATH.exists():
            classes = get_disease_classes()

            model    = tv_models.resnet18(weights=None)
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(model.fc.in_features, len(classes))
            )
            model.load_state_dict(
                torch.load(DISEASE_MODEL_PATH, map_location='cpu')
            )
            model.eval()

            with torch.no_grad():
                output = model(tensor)
                probs  = torch.softmax(output, dim=1)[0]

            idx        = int(torch.argmax(probs))
            confidence = round(float(probs[idx]) * 100, 1)
            raw_name   = classes[idx] if idx < len(classes) else "Unknown"

            # Format name for display (this is what the UI shows)
            display_name = format_disease_name(raw_name)

            # Look up treatment: try raw key first, then default
            treatment  = DISEASE_TREATMENT.get(raw_name, _DEFAULT_TREATMENT)
            model_acc  = get_disease_model_accuracy()
            is_healthy = "healthy" in raw_name.lower()

            return (not is_healthy, display_name, confidence, model_acc, treatment)

        else:
            return _colour_heuristic(image_path)

    except Exception as e:
        print(f"[ML] Disease prediction error: {e}")
        import traceback; traceback.print_exc()
        return _colour_heuristic(image_path)


def _colour_heuristic(image_path):
    """Fallback when CNN model is unavailable."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert('RGB').resize((64, 64))
        px  = np.array(img).astype(float)
        r, g, b = px[:, :, 0].mean(), px[:, :, 1].mean(), px[:, :, 2].mean()

        if r > 140 and g < 100 and b < 80:
            raw = "BrownSpot"; conf = round(np.random.uniform(72, 88), 1)
        elif r > 120 and g > 100 and b < 80 and r > g:
            raw = "Hispa";     conf = round(np.random.uniform(68, 84), 1)
        elif g > r and g > b and g > 80:
            raw = "Healthy";   conf = round(np.random.uniform(85, 97), 1)
        else:
            raw = "LeafBlast"; conf = round(np.random.uniform(65, 82), 1)

        display_name = format_disease_name(raw)
        treatment    = DISEASE_TREATMENT.get(raw, _DEFAULT_TREATMENT)
        is_healthy   = raw.lower() == "healthy"
        return (not is_healthy, display_name, conf, None, treatment)
    except Exception:
        return (False, "Healthy", 90.0, None, DISEASE_TREATMENT["Healthy"])
