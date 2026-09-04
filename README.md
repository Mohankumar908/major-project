# Smart Agriculture System — Setup Guide

## Quick Start (Run today)

### 1. Install dependencies
```bash
pip install django djangorestframework django-cors-headers Pillow scikit-learn numpy joblib
# For disease detection (optional, heavy):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Setup Django
```bash
cd backend/smart_agri      # or wherever you put this project
python manage.py makemigrations agriculture
python manage.py migrate
python manage.py createsuperuser   # optional for admin panel
python manage.py runserver 0.0.0.0:8000
```

### 3. Open Dashboard
- Browser: http://localhost:8000
- Admin:   http://localhost:8000/admin

---

## API Endpoints

| Method | URL | Purpose |
|--------|-----|---------|
| POST | `/api/sensor-data` | ESP32 sends sensor readings |
| GET  | `/api/get-data`    | Dashboard polls latest data |
| GET  | `/api/predict-growth?ph=6.5&n=80&p=40&k=60&temp=28&humidity=65&moisture=45` | Manual prediction |
| POST | `/api/predict-disease` | Upload plant image for CNN |
| GET  | `/api/recent-images`  | Get recent image predictions |

---

## ESP32 Setup
1. Open `esp32_sensor_node.ino` in Arduino IDE
2. Set your WiFi credentials
3. Set `SERVER_URL` to your PC's local IP (run `hostname -I` on Linux / `ipconfig` on Windows)
4. Upload to ESP32

---

## Testing Without Hardware (Demo Mode)
Use this curl command to simulate ESP32 data:
```bash
curl -X POST http://localhost:8000/api/sensor-data \
  -H "Content-Type: application/json" \
  -d '{"ph":6.5,"npk_n":80,"npk_p":40,"npk_k":60,"temperature":28,"humidity":65,"soil_moisture":45,"device_id":"ESP32_01"}'
```

Or run the Python simulator:
```bash
python simulate_sensors.py
```

---

## Adding Your Trained Disease Model
1. Train your CNN on PlantVillage dataset    python train_disease_model.py --data dataset --epochs 30
2. Save as `ml_models/disease_model.pth`
3. Update `DISEASE_CLASSES` list in `agriculture/ml_utils.py` to match your training labels
4. Restart Django — it auto-loads the model

---

## Project Structure
```
smart_agri/
├── manage.py
├── requirements.txt
├── esp32_sensor_node.ino       ← Upload to ESP32
├── simulate_sensors.py         ← Test without hardware
├── smart_agri/
│   ├── settings.py
│   └── urls.py
├── agriculture/
│   ├── models.py               ← SensorData, GrowthPrediction, PlantImage
│   ├── views.py                ← All API endpoints + dashboard view
│   ├── ml_utils.py             ← Growth model + disease detection
│   └── templates/agriculture/dashboard.html
├── ml_models/                  ← Auto-created, stores .pkl and .pth files
└── media/plant_images/         ← Uploaded images stored here
```
