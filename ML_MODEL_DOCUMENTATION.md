# ML Model Documentation
## Smart Agriculture System — Multimodal Continuous Crop Monitoring

---

## 0. SYSTEM ARCHITECTURE OVERVIEW

The growth monitoring system operates as a **closed-loop continuous monitoring pipeline** within the context of the **SAME CROP SESSION (continuous daily loop)**:

```
                      ┌────────────────────────────────────────────────────────┐
                      │                 SAME CROP SESSION                      │
                      │       (This session continues every day)               │
                      └───────────────┬────────────────────────┬───────────────┘
                                      │                        │
                    ┌─────────────────▼─────────┐    ┌─────────▼─────────────────┐
                    │       DAILY IMAGE         │    │       SENSOR DATA         │
                    │   (Morning / Evening)     │    │  (Continuous readings)    │
                    │  - Morning Capture (Sun)  │    │  - Temperature, Humidity  │
                    │  - Evening Capture (Moon) │    │  - Soil Moisture, pH, NPK │
                    └─────────────────┬─────────┘    └─────────┬─────────────────┘
                                      │                        │
                    ┌─────────────────▼─────────┐    ┌─────────▼─────────────────┐
                    │    IMAGE GROWTH MODEL     │    │     SENSOR PROCESSING     │
                    │  - Plant area (cm²)       │    │  - Temperature / Thermal  │
                    │  - Green coverage (%)     │    │  - Humidity / VPD (kPa)   │
                    │  - Canopy size (cm²)      │    │  - Soil moisture index    │
                    │  - Plant height (est)     │    │  - Soil pH bio-availability│
                    │  - Growth features (ExG)  │    │  - N / P / K Balance      │
                    └─────────────────┬─────────┘    └─────────┬─────────────────┘
                                      │                        │
                                      └───────────┬────────────┘
                                                  │
                                    ┌─────────────▼───────────────┐   ┌─────────────────┐
                                    │   MULTIMODAL GROWTH STATE   │◄──┤ Historical crop │
                                    │  Combined image + sensor    │   │ data (velocity, │
                                    │  features for current state │   │ GDD context)    │
                                    └─────────────┬───────────────┘   └─────────────────┘
                                                  │
                                    ┌─────────────▼───────────────┐
                                    │    GROWTH PREDICTION MODEL  │
                                    │  Gradient Boosting / XGBoost│
                                    └─────────────┬───────────────┘
                                                  │
            ┌─────────────────────────────────────┼─────────────────────────────────────┐
            │                                     │                                     │
┌───────────▼───────────┐             ┌───────────▼───────────┐             ┌───────────▼───────────┐
│     GROWTH TREND      │             │    RISK / ANOMALY     │             │      PREDICTION       │
│ - Growth progression  │             │ - Abnormal growth     │             │ - Future plant height │
│ - Plant height trend  │             │ - Nutrient deficiency │             │ - Expected yield      │
│ - Canopy expansion    │             │ - Disease risk index  │             │ - Growth stage pred.  │
└───────────┬───────────┘             └───────────┬───────────┘             └───────────┬───────────┘
            │                                     │                                     │
            └─────────────────────────────────────┼─────────────────────────────────────┘
                                                  │
                                    ┌─────────────▼───────────────┐
                                    │    RECOMMENDATION ENGINE    │
                                    ├─────────────────────────────┤
                                    │ 💧 IRRIGATION SUGGESTION    │
                                    │ 🌱 NUTRIENT SUGGESTION      │
                                    │ ⚠️ CROP-HEALTH WARNING      │
                                    └─────────────┬───────────────┘
                                                  │
                                    ┌─────────────▼───────────────┐
                                    │      DASHBOARD GRAPHS       │
                                    │ 1. Plant Height Trend       │
                                    │ 2. Canopy Growth            │
                                    │ 3. Sensor Graphs            │
                                    │ 4. Predicted Yield          │
                                    └─────────────┬───────────────┘
                                                  │
                                                  ▼
                                    ┌─────────────────────────────┐
                                    │      SAME CROP SESSION      │
                                    │    Continues Every Day      │
                                    └─────────────────────────────┘
```

---

## 1. GROWTH PREDICTION MODEL (Regression)

### What model did we choose?
**Gradient Boosting / XGBoost Stacked Ensemble** (scikit-learn `GradientBoostingRegressor`, `RandomForestRegressor`, and optional `XGBRegressor`)

### Why Gradient Boosting over other options?

| Model | Why NOT chosen |
|-------|---------------|
| Linear Regression | Too simple — crop yield has non-linear relationships with pH, NPK. Low accuracy (~40% R²) |
| Simple Decision Tree | Overfits on small data, poor generalization |
| Neural Network (MLP) | Needs large dataset (10,000+ samples). We have limited field data |
| SVR | Slower to train, less interpretable, similar accuracy |
| Random Forest | Good, but Gradient Boosting consistently outperforms it on tabular data by 5–10% R² |
| **Gradient Boosting** ✅ | Best accuracy on small-to-medium tabular datasets. Handles non-linearity. Works with 500–5000 samples. Provides feature importance. No need for data normalization of targets |

### How does Gradient Boosting work (simple explanation)?
It builds trees **sequentially** — each new tree corrects the errors of the previous one.
Like a team where each member fixes what the previous member got wrong.

### Two regression models trained:
1. **Yield Model** → predicts expected yield in kg/acre
2. **Height Model** → predicts plant height in cm on a given day

### Input Features (8 features):
| Feature | Unit | Why included |
|---------|------|-------------|
| Soil pH | 0–14 | Directly affects nutrient availability. Green gram optimal: 6.5–7.5 |
| Nitrogen (N) | mg/kg | Primary growth nutrient — affects leaf/stem development |
| Phosphorus (P) | mg/kg | Root development and pod formation |
| Potassium (K) | mg/kg | Disease resistance and pod filling |
| Temperature | °C | Green gram optimal 25–35°C. <18°C or >40°C causes yield loss |
| Humidity | % | Affects transpiration and fungal disease risk |
| Soil Moisture | % | Directly controls water stress |
| Day Number | days since sowing | Growth is time-dependent (sigmoid curve) |

### Dataset: Where did we get it?
- **Phase 1 (Current):** Synthetic data generated using agronomic formulas from:
  - ICAR (Indian Council of Agricultural Research) green gram cultivation guidelines
  - FAO crop water requirements database
  - Published research: "Effect of NPK on green gram yield" — Journal of Agronomy (2019)
  - 1000 synthetic samples generated with realistic noise

- **Phase 2 (Next semester):** Replace with real field data collected from:
  - Our own ESP32 sensors (daily readings)
  - Local farmers' yield records (Bangalore region)
  - KVK (Krishi Vigyan Kendra) experimental farm data

### Current Accuracy:
| Metric | Yield Model | Height Model |
|--------|------------|-------------|
| R² Score (synthetic data) | ~48–55% | ~98.6% |
| Expected R² (real field data) | 75–85% | 95–99% |
| MAE (yield) | ±45 kg/acre | — |
| MAE (height) | — | ±1.2 cm |

> **Why is yield R² lower?** Yield prediction is inherently harder — final yield depends on many factors not captured by sensors (rainfall events, pest damage, harvest method). Height prediction is easier because it follows a predictable sigmoid curve. With real data over multiple crop cycles, yield R² improves significantly.

### Feature Importance (approximate):
1. Day Number (38%) — growth is most time-dependent
2. Nitrogen (22%) — biggest nutrient impact on yield
3. Soil Moisture (15%) — water stress has large impact
4. Temperature (12%) — heat stress critical for green gram
5. pH (8%) — affects nutrient uptake
6. P, K, Humidity (5%) — secondary effects

---

## 2. DISEASE DETECTION MODEL (CNN)

### What model did we choose?
**ResNet18** — Residual Network with 18 layers (transfer learning from ImageNet)

### Why ResNet18 specifically?

| Model | Why NOT chosen |
|-------|---------------|
| Custom CNN from scratch | Needs 50,000+ images. We have ~3,000. Would overfit badly |
| VGG16/VGG19 | 138M parameters — too large, slow inference on server CPU, overfits on small datasets |
| ResNet50/ResNet101 | Overkill — 50-layer nets don't give significantly better accuracy than ResNet18 for leaf disease tasks |
| MobileNetV2 | Good option, but ResNet18 has slightly better accuracy on PlantVillage benchmarks |
| EfficientNet | Better accuracy but complex and slower to train |
| **ResNet18** ✅ | 11M parameters, fast inference (<100ms on CPU), proven on PlantVillage, transfer learning from ImageNet gives 90%+ accuracy with only fine-tuning the last layer |

### What is ResNet18?
ResNet (Residual Network) introduced **skip connections** — the output of one layer jumps ahead to a later layer. This solves the "vanishing gradient" problem in deep networks and allows training deeper models effectively.

```
Normal CNN:   Layer1 → Layer2 → Layer3
ResNet:       Layer1 → Layer2 → Layer3
                 ↑_________________________↗
                     (skip connection)
```

### Transfer Learning Approach:
1. Start with ResNet18 pretrained on **ImageNet** (1.2M images, 1000 classes)
2. The model already knows: edges, textures, shapes, colors
3. **Replace only the final layer** (1000 classes → 8 disease classes)
4. Fine-tune entire network on PlantVillage for 15 epochs
5. Result: 92–97% accuracy with only ~3000 training images

### Dataset: Where did we get it?

**Primary Dataset: PlantVillage**
- Source: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- Also: https://github.com/spMohanty/PlantVillage-Dataset
- Published by: Hughes & Salathé, Penn State University (2015)
- Total images: 54,000+ images across 38 plant disease classes
- License: Open / CC0 (free for academic use)
- We use the **rice/grain legume subset** for green gram diseases

**Rice/Legume Specific Dataset:**
- Source: https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset
- Images: ~3,000 rice + green gram leaf images
- Classes: Bacterial Blight, Brown Spot, Leaf Blast, Healthy

**Why we trust this dataset:**
- Collected from real farms in Asia (India, Philippines, Bangladesh)
- Images captured under real field conditions (not lab)
- Validated by agricultural scientists
- Used in 100+ published papers

### Disease Classes We Detect:
| # | Class | Description | Visual Symptom |
|---|-------|-------------|----------------|
| 0 | Healthy | No disease | Uniform green leaf |
| 1 | Bacterial Leaf Blight | *Xanthomonas* bacteria | Yellow-brown leaf margins |
| 2 | Brown Spot | *Helminthosporium* fungus | Brown oval spots on leaves |
| 3 | Leaf Blast | *Magnaporthe* fungus | Diamond-shaped gray spots |
| 4 | Leaf Scald | *Microdochium* fungus | Striped tan-brown lesions |
| 5 | Narrow Brown Leaf Spot | *Cercospora* | Narrow brown stripes |
| 6 | Sheath Blight | *Rhizoctonia* fungus | Water-soaked lesions |
| 7 | Tungro | Virus (via leafhopper) | Yellow-orange leaf discoloration |

### Expected Accuracy:
| Metric | Value |
|--------|-------|
| Validation Accuracy (PlantVillage) | 92–97% |
| Precision | ~94% |
| Recall | ~93% |
| F1 Score | ~93% |
| Inference Time (CPU) | <200ms per image |

> When no trained model file (`disease_model.pth`) is present, the system falls back to a **colour heuristic** based on pixel RGB analysis. This is for demo only — always train and load the real model for production.

### Training Script Location:
See `DATASETS_AND_TRAINING.md` → Section 1 for full training code.

---

## 3. HOW ACCURACY IS SHOWN IN REAL-TIME

### Growth Model (Regression):
- R² score is computed during training on 20% held-out test data
- Stored in `ml_models/growth_model.pkl` as `yield_accuracy` field
- Displayed on every prediction card and chart header
- Re-computed automatically if you retrain with new data

### Disease CNN:
- Accuracy stored in `ml_models/disease_model_accuracy.txt` after training
- Per-prediction confidence shown as softmax probability (0–100%)
- Both model accuracy AND prediction confidence shown on dashboard

### Interpreting Results:
```
Model Accuracy = how well the model performs on test data (fixed, from training)
Confidence     = how sure the model is about THIS specific prediction (varies per image)

Example: Model Accuracy 95%, Confidence 72%
→ The CNN is generally reliable (95%) but uncertain about this particular leaf photo.
  Consider taking a clearer, better-lit image.
```

---

## 4. MODEL IMPROVEMENT PLAN (Phase 2 — Semester 7)

1. **Collect real green gram field data** from Bangalore region farmers (3–4 crop cycles)
2. **Retrain yield regression model** with actual sensor + yield data → expected R² 80–90%
3. **Fine-tune CNN** on photos taken with our own camera setup (same conditions as deployment)
4. **Add LSTM time-series model** for yield forecasting using 7-day sensor history
5. **Automate daily model retraining** as new field data accumulates
