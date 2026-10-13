import json
import math
from datetime import date
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.db.models import Avg, Count, Max

from django.conf import settings
from .models import (
    SensorData, GrowthPrediction, PlantImage,
    CropSession, DailyGrowthRecord, CropNote,
)
from .ml_utils import (
    predict_growth, predict_disease,
    extract_greenness, extract_leaf_area, extract_image_features,
    extract_image_growth_model, process_sensor_telemetry,
    build_multimodal_growth_state, run_growth_prediction_model,
    run_recommendation_engine,
    format_disease_name, calculate_agronomic_advisory,
)
from .auto_retrain import maybe_trigger_retrain


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

CROP_ICONS = {
    'Green Gram': '🫘', 'Rice': '🌾', 'Wheat': '🌿', 'Tomato': '🍅',
    'Maize': '🌽',      'Potato': '🥔', 'Soybean': '🫘', 'Sugarcane': '🎋',
    'Cotton': '☁️',     'Other': '🌱',
}


def _session_summary(session):
    latest_rec    = (
        session.growth_records
        .exclude(health_score__isnull=True)
        .order_by('-date', '-slot')
        .first()
    )
    latest_sensor = session.sensor_readings.order_by('-timestamp').first()
    return {
        'id':            session.id,
        'crop_name':     session.crop_name,
        'crop_variety':  session.crop_variety,
        'field_name':    session.field_name,
        'field_acres':   session.field_acres,
        'farmer_name':   session.farmer_name,
        'location':      session.location,
        'sowing_date':   str(session.sowing_date),
        'expected_harvest': str(session.expected_harvest) if session.expected_harvest else None,
        'status':        session.status,
        'days':          session.days_since_sowing,
        'progress_pct':  session.progress_pct,
        'health_score':  latest_rec.health_score  if latest_rec else None,
        'growth_stage':  latest_rec.growth_stage  if latest_rec else None,
        'yield_pred':    latest_rec.predicted_yield_per_acre if latest_rec else None,
        'records_count': session.growth_records.count(),
        'notes_count':   session.crop_notes.count(),
        'icon':          CROP_ICONS.get(session.crop_name, '🌱'),
        'latest_sensor': {
            'ph':            latest_sensor.ph,
            'temperature':   latest_sensor.temperature,
            'humidity':      latest_sensor.humidity,
            'soil_moisture': latest_sensor.soil_moisture,
            'timestamp':     latest_sensor.timestamp.strftime('%Y-%m-%d %H:%M'),
        } if latest_sensor else None,
    }


def _ideal_curve(total_days=60):
    """Sigmoid height + ramp yield ideal baseline."""
    ideal_days, ideal_heights, ideal_yields = [], [], []
    for d in range(1, total_days + 1):
        h = round(4.0 + 46.0 * (1 / (1 + math.exp(-0.13 * (d - 26)))), 1)
        y = round(200.0 + 400.0 * (d / total_days) ** 1.3, 1)
        ideal_days.append(d)
        ideal_heights.append(h)
        ideal_yields.append(y)
    return ideal_days, ideal_heights, ideal_yields


# ═══════════════════════════════════════════════════════════════════════════
# PAGE VIEWS
# ═══════════════════════════════════════════════════════════════════════════

def home(request):
    active_sessions    = CropSession.objects.filter(status='active').order_by('-created_at')
    completed_sessions = CropSession.objects.filter(status='completed').order_by('-created_at')
    paused_sessions    = CropSession.objects.filter(status='paused').order_by('-created_at')
    total_active       = active_sessions.count()
    total_completed    = completed_sessions.count()
    disease_alerts     = PlantImage.objects.filter(processed=True, disease_detected=True).count()
    return render(request, 'agriculture/home.html', {
        'active_sessions':    active_sessions,
        'completed_sessions': completed_sessions,
        'paused_sessions':    paused_sessions,
        'total_active':       total_active,
        'total_completed':    total_completed,
        'disease_alerts':     disease_alerts,
        'crop_icons':         CROP_ICONS,
    })


def session_detail(request, session_id):
    session       = get_object_or_404(CropSession, id=session_id)
    latest_sensor = session.sensor_readings.order_by('-timestamp').first()
    latest_rec    = (
        session.growth_records
        .exclude(health_score__isnull=True)
        .order_by('-date', '-slot')
        .first()
    )

    # Calculate or retrieve current multimodal state & predictions
    latest_img_rec = (
        session.growth_records
        .exclude(actual_greenness_score__isnull=True)
        .order_by('-date', '-slot')
        .first()
    )

    img_features_for_pred = None
    if latest_img_rec:
        img_features_for_pred = {
            'plant_area_px': 25000,
            'plant_area_cm2': latest_img_rec.actual_leaf_area_cm2 or 300.0,
            'green_coverage_percent': latest_img_rec.actual_leaf_coverage_percent or 40.0,
            'canopy_size_cm2': latest_img_rec.canopy_size_cm2 or 360.0,
            'estimated_plant_height_cm': latest_img_rec.estimated_plant_height_cm or latest_img_rec.actual_height_cm or 24.0,
            'growth_features': latest_img_rec.growth_features or {
                'exg_greenness': latest_img_rec.actual_greenness_score or 55.0,
                'foliage_density': 0.72,
                'aspect_ratio': 1.10,
                'leaf_clusters': 5,
                'canopy_spread_width_cm': 18.0,
            },
            'image_features_used': True,
        }

    # Evaluate complete pipeline
    pipeline_result = predict_growth(
        ph          = latest_sensor.ph            if latest_sensor and latest_sensor.ph is not None else 6.6,
        nitrogen    = latest_sensor.npk_nitrogen   if latest_sensor and latest_sensor.npk_nitrogen is not None else 62.0,
        phosphorus  = latest_sensor.npk_phosphorus if latest_sensor and latest_sensor.npk_phosphorus is not None else 32.0,
        potassium   = latest_sensor.npk_potassium  if latest_sensor and latest_sensor.npk_potassium is not None else 52.0,
        temperature = latest_sensor.temperature    if latest_sensor and latest_sensor.temperature is not None else 28.0,
        humidity    = latest_sensor.humidity       if latest_sensor and latest_sensor.humidity is not None else 64.0,
        moisture    = latest_sensor.soil_moisture  if latest_sensor and latest_sensor.soil_moisture is not None else 48.0,
        day_number  = session.days_since_sowing,
        field_acres = session.field_acres,
        session     = session,
        image_features = img_features_for_pred,
    )

    growth_trend         = pipeline_result['growth_trend']
    risk_anomaly         = pipeline_result['risk_anomaly']
    prediction_data      = pipeline_result['prediction']
    irrigation_sugg      = pipeline_result['irrigation_suggestion']
    nutrient_sugg        = pipeline_result['nutrient_suggestion']
    crop_health_warn     = pipeline_result['crop_health_warning']
    multimodal_state     = pipeline_result['multimodal_state']
    sensor_stream        = multimodal_state['sensor_stream']
    image_stream         = multimodal_state['image_stream']

    # Chart bootstrap data (4 DASHBOARD GRAPHS)
    growth_records = session.growth_records.order_by('day_number', 'slot')[:90]
    chart_days, actual_heights, pred_heights, pred_yields = [], [], [], []
    actual_greenness, actual_leaf_cov, canopy_sizes, expansion_velocities = [], [], [], []
    seen = {}
    prev_cov = None

    for r in growth_records:
        if r.day_number not in seen:
            seen[r.day_number] = True
            chart_days.append(r.day_number)
            actual_heights.append(r.actual_height_cm)
            pred_heights.append(r.predicted_height_cm or r.actual_height_cm)
            pred_yields.append(r.predicted_yield_per_acre or pipeline_result['predicted_yield_per_acre'])
            actual_greenness.append(r.actual_greenness_score)
            cov = r.actual_leaf_coverage_percent or 35.0
            actual_leaf_cov.append(cov)
            canopy_sizes.append(r.canopy_size_cm2 or round(cov * 9.0, 1))

            if prev_cov is not None:
                expansion_velocities.append(round(cov - prev_cov, 1))
            else:
                expansion_velocities.append(round(cov * 0.1, 1))
            prev_cov = cov

    # Default seed points if session just started
    if not chart_days:
        chart_days = [max(1, session.days_since_sowing)]
        actual_heights = [pipeline_result['predicted_height_cm']]
        pred_heights = [pipeline_result['predicted_height_cm']]
        pred_yields = [pipeline_result['predicted_yield_per_acre']]
        actual_leaf_cov = [image_stream['green_coverage_percent']]
        canopy_sizes = [image_stream.get('canopy_size_cm2', 360.0)]
        expansion_velocities = [2.2]

    total_days = 60
    if session.expected_harvest:
        total_days = max(30, (session.expected_harvest - session.sowing_date).days)
    ideal_days, ideal_heights, ideal_yields = _ideal_curve(total_days)

    # Sensor history sparkline & multi-series chart
    sensor_history = list(
        session.sensor_readings
        .values('timestamp', 'ph', 'temperature', 'humidity', 'soil_moisture', 'npk_nitrogen', 'npk_phosphorus', 'npk_potassium')
        [:25]
    )
    for s in sensor_history:
        s['time_label'] = s['timestamp'].strftime('%H:%M')

    today        = date.today()
    morning_done = session.growth_records.filter(date=today, slot='morning', image_uploaded=True).exists()
    evening_done = session.growth_records.filter(date=today, slot='evening', image_uploaded=True).exists()
    recent_notes = session.crop_notes.order_by('-created_at')[:5]
    recent_images = (
        PlantImage.objects.filter(
            dailygrowthrecord__session=session, processed=True,
        ).distinct().order_by('-uploaded_at')[:8]
    )
    avg_health = (
        session.growth_records
        .exclude(health_score__isnull=True)
        .order_by('-date')[:7]
        .aggregate(avg=Avg('health_score'))['avg']
    ) or pipeline_result['health_score']

    advisories = calculate_agronomic_advisory(
        sensor_stream['ph'], sensor_stream['npk_nitrogen'],
        sensor_stream['npk_phosphorus'], sensor_stream['npk_potassium'],
        sensor_stream['temperature'], sensor_stream['humidity'],
        sensor_stream['soil_moisture']
    )

    all_growth_records = session.growth_records.order_by('-date', '-slot')[:50]
    all_sensor_records = session.sensor_readings.order_by('-timestamp')[:50]

    return render(request, 'agriculture/session_detail.html', {
        'session':                   session,
        'latest_sensor':             latest_sensor,
        'latest_rec':                latest_rec,
        'sensor_stream':             sensor_stream,
        'image_stream':              image_stream,
        'multimodal_state':          multimodal_state,
        'growth_trend':              growth_trend,
        'risk_anomaly':              risk_anomaly,
        'prediction_data':           prediction_data,
        'irrigation_sugg':           irrigation_sugg,
        'nutrient_sugg':             nutrient_sugg,
        'crop_health_warn':          crop_health_warn,
        'morning_done':              morning_done,
        'evening_done':              evening_done,
        'recent_notes':              recent_notes,
        'recent_images':             recent_images,
        'advisories':                advisories,
        'all_growth_records':        all_growth_records,
        'all_sensor_records':        all_sensor_records,
        'avg_health':                round(avg_health, 1) if avg_health else 90.0,
        'icon':                      CROP_ICONS.get(session.crop_name, '🌱'),
        # JSON bootstrap for all 4 dashboard graphs
        'chart_days_json':           json.dumps(chart_days),
        'actual_heights_json':       json.dumps(actual_heights),
        'pred_heights_json':         json.dumps(pred_heights),
        'ideal_heights_json':        json.dumps(ideal_heights[:len(chart_days)] if len(ideal_heights) >= len(chart_days) else ideal_heights),
        'actual_leaf_cov_json':      json.dumps(actual_leaf_cov),
        'canopy_sizes_json':         json.dumps(canopy_sizes),
        'expansion_velocities_json': json.dumps(expansion_velocities),
        'pred_yields_json':          json.dumps(pred_yields),
        'ideal_yields_json':         json.dumps(ideal_yields[:len(chart_days)] if len(ideal_yields) >= len(chart_days) else ideal_yields),
        'sensor_history_json':       json.dumps(list(reversed(sensor_history)), default=str),
    })


def disease_detection(request):
    recent       = PlantImage.objects.filter(processed=True).order_by('-uploaded_at')[:20]
    total_scans  = PlantImage.objects.filter(processed=True).count()
    disease_found = PlantImage.objects.filter(processed=True, disease_detected=True).count()
    healthy_found = PlantImage.objects.filter(processed=True, disease_detected=False).count()
    return render(request, 'agriculture/disease_detection.html', {
        'recent_images': recent,
        'total_scans':   total_scans,
        'disease_found': disease_found,
        'healthy_found': healthy_found,
    })


# ═══════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT APIs
# ═══════════════════════════════════════════════════════════════════════════

@require_http_methods(["GET"])
def api_sessions(request):
    sessions = CropSession.objects.all()
    return JsonResponse({'sessions': [_session_summary(s) for s in sessions]})


@csrf_exempt
@require_http_methods(["POST"])
def api_create_session(request):
    try:
        body    = json.loads(request.body)
        sowing_raw = body.get('sowing_date') or date.today().isoformat()
        # Parse string → date object so days_since_sowing arithmetic works
        from datetime import datetime as _dt
        if isinstance(sowing_raw, str):
            sowing = _dt.strptime(sowing_raw[:10], '%Y-%m-%d').date()
        else:
            sowing = sowing_raw

        harvest_raw = body.get('expected_harvest') or None
        if isinstance(harvest_raw, str) and harvest_raw:
            harvest = _dt.strptime(harvest_raw[:10], '%Y-%m-%d').date()
        else:
            harvest = None

        session = CropSession.objects.create(
            crop_name        = body.get('crop_name', 'Green Gram'),
            crop_variety     = body.get('crop_variety', ''),
            field_name       = body.get('field_name', ''),
            field_acres      = float(body.get('field_acres', 1.0)),
            farmer_name      = body.get('farmer_name', ''),
            location         = body.get('location', ''),
            sowing_date      = sowing,
            expected_harvest = harvest,
            notes            = body.get('notes', ''),
            status           = 'active',
        )
        return JsonResponse({'status': 'ok', 'session': _session_summary(session)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def api_close_session(request, session_id):
    session = get_object_or_404(CropSession, id=session_id)
    session.status = 'completed'
    session.save()
    return JsonResponse({'status': 'ok'})


@csrf_exempt
@require_http_methods(["POST"])
def api_delete_session(request, session_id):
    session = get_object_or_404(CropSession, id=session_id)
    session.delete()
    return JsonResponse({'status': 'ok'})


@csrf_exempt
@require_http_methods(["POST"])
def api_simulate_sensor(request, session_id):
    """
    Generate realistic simulated IoT telemetry for demonstration/testing.
    Feeds telemetry through the growth prediction pipeline and returns updated readings & advisories.
    """
    session = get_object_or_404(CropSession, id=session_id)
    try:
        import random
        ph = round(random.uniform(6.4, 6.9), 1)
        n = round(random.uniform(55, 75), 1)
        p = round(random.uniform(28, 42), 1)
        k = round(random.uniform(48, 65), 1)
        temp = round(random.uniform(26.0, 30.5), 1)
        hum = round(random.uniform(58.0, 68.0), 1)
        moist = round(random.uniform(46.0, 56.0), 1)

        if request.body:
            try:
                body = json.loads(request.body)
                if 'ph' in body and body['ph'] is not None: ph = float(body['ph'])
                if 'npk_n' in body and body['npk_n'] is not None: n = float(body['npk_n'])
                if 'npk_p' in body and body['npk_p'] is not None: p = float(body['npk_p'])
                if 'npk_k' in body and body['npk_k'] is not None: k = float(body['npk_k'])
                if 'temperature' in body and body['temperature'] is not None: temp = float(body['temperature'])
                if 'humidity' in body and body['humidity'] is not None: hum = float(body['humidity'])
                if 'soil_moisture' in body and body['soil_moisture'] is not None: moist = float(body['soil_moisture'])
            except Exception:
                pass

        sensor = SensorData.objects.create(
            session=session,
            ph=ph,
            npk_nitrogen=n,
            npk_phosphorus=p,
            npk_potassium=k,
            temperature=temp,
            humidity=hum,
            soil_moisture=moist,
            device_id='ESP32_SIM',
        )

        growth_res = predict_growth(
            ph=ph, nitrogen=n, phosphorus=p, potassium=k,
            temperature=temp, humidity=hum, moisture=moist,
            day_number=session.days_since_sowing,
            field_acres=session.field_acres,
            session=session,
        )

        GrowthPrediction.objects.create(
            sensor_data              = sensor,
            predicted_yield_per_acre = growth_res['predicted_yield_per_acre'],
            growth_stage             = growth_res['growth_stage'],
            health_score             = growth_res['health_score'],
            recommendation           = growth_res['recommendation_text'],
            model_accuracy           = growth_res['model_accuracy'],
        )

        today = date.today()
        slot = 'morning' if timezone.now().hour < 14 else 'evening'
        rec, _ = DailyGrowthRecord.objects.get_or_create(
            session=session, date=today, slot=slot,
            defaults={'day_number': session.days_since_sowing},
        )
        if not rec.image_uploaded:
            rec.predicted_height_cm      = growth_res['predicted_height_cm']
            rec.predicted_yield_per_acre = growth_res['predicted_yield_per_acre']
            rec.health_score             = growth_res['health_score']
            rec.growth_stage             = growth_res['growth_stage']
            rec.recommendation           = growth_res['recommendation_text']
            rec.model_accuracy           = growth_res['model_accuracy']
            rec.sensor_data              = sensor
            rec.multimodal_state         = growth_res['multimodal_state']
            rec.growth_trend_data        = growth_res['growth_trend']
            rec.risk_anomaly_data        = growth_res['risk_anomaly']
            rec.prediction_data          = growth_res['prediction']
            rec.irrigation_suggestion    = growth_res['irrigation_suggestion']
            rec.nutrient_suggestion      = growth_res['nutrient_suggestion']
            rec.crop_health_warning      = growth_res['crop_health_warning']
            rec.save()

        advisories = calculate_agronomic_advisory(ph, n, p, k, temp, hum, moist)

        return JsonResponse({
            'status': 'ok',
            'sensor': {
                'id':            sensor.id,
                'ph':            sensor.ph,
                'npk_n':         sensor.npk_nitrogen,
                'npk_p':         sensor.npk_phosphorus,
                'npk_k':         sensor.npk_potassium,
                'temperature':   sensor.temperature,
                'humidity':      sensor.humidity,
                'soil_moisture': sensor.soil_moisture,
                'timestamp':     sensor.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'device_id':     sensor.device_id,
            },
            'sensor_stream':        growth_res['multimodal_state']['sensor_stream'],
            'image_stream':         growth_res['multimodal_state']['image_stream'],
            'multimodal_state':     growth_res['multimodal_state'],
            'growth_trend':         growth_res['growth_trend'],
            'risk_anomaly':         growth_res['risk_anomaly'],
            'prediction':           growth_res['prediction'],
            'irrigation_suggestion': growth_res['irrigation_suggestion'],
            'nutrient_suggestion':  growth_res['nutrient_suggestion'],
            'crop_health_warning':  growth_res['crop_health_warning'],
            'advisories':           advisories,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@require_http_methods(["GET"])
def api_export_session_csv(request, session_id):
    """Export complete crop telemetry & growth data as a structured CSV report."""
    import csv
    from django.http import HttpResponse
    session = get_object_or_404(CropSession, id=session_id)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="crop_report_{session.crop_name}_{session.id}.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'Crop Name', 'Variety', 'Field Name', 'Acres', 'Farmer',
        'Sowing Date', 'Days Since Sowing', 'Date', 'Slot',
        'Observed Height (cm)', 'Predicted Height (cm)',
        'Predicted Yield (kg/acre)', 'Health Score (%)', 'Growth Stage',
        'Greenness Score (ExG)', 'Leaf Coverage (%)', 'Leaf Area (cm2)',
        'Recommendation'
    ])

    records = session.growth_records.order_by('date', 'slot')
    for r in records:
        writer.writerow([
            session.crop_name, session.crop_variety, session.field_name, session.field_acres, session.farmer_name,
            session.sowing_date, r.day_number, r.date, r.slot,
            r.actual_height_cm, r.predicted_height_cm,
            r.predicted_yield_per_acre, r.health_score, r.growth_stage,
            r.actual_greenness_score, r.actual_leaf_coverage_percent, r.actual_leaf_area_cm2,
            r.recommendation
        ])
    return response


# ═══════════════════════════════════════════════════════════════════════════
# GROWTH MONITORING APIs
# ═══════════════════════════════════════════════════════════════════════════

@csrf_exempt
@require_http_methods(["POST"])
def api_sensor_data(request, session_id):
    """Receive IoT sensor payload and run growth prediction (sensor-only path)."""
    session = get_object_or_404(CropSession, id=session_id)
    try:
        body   = json.loads(request.body)
        sensor = SensorData.objects.create(
            session        = session,
            ph             = body.get('ph'),
            npk_nitrogen   = body.get('npk_n')   or body.get('nitrogen'),
            npk_phosphorus = body.get('npk_p')   or body.get('phosphorus'),
            npk_potassium  = body.get('npk_k')   or body.get('potassium'),
            temperature    = body.get('temperature'),
            humidity       = body.get('humidity'),
            soil_moisture  = body.get('soil_moisture') or body.get('moisture'),
            device_id      = body.get('device_id', 'ESP32_01'),
        )

        # Pull latest image metrics to feed into the Multimodal State Fusion
        latest_img_rec = (
            session.growth_records
            .exclude(actual_greenness_score__isnull=True)
            .order_by('-date', '-slot')
            .first()
        )
        img_feats = None
        if latest_img_rec:
            img_feats = {
                'plant_area_px': 25000,
                'plant_area_cm2': latest_img_rec.actual_leaf_area_cm2 or 300.0,
                'green_coverage_percent': latest_img_rec.actual_leaf_coverage_percent or 40.0,
                'canopy_size_cm2': latest_img_rec.canopy_size_cm2 or 360.0,
                'estimated_plant_height_cm': latest_img_rec.estimated_plant_height_cm or 24.0,
                'growth_features': latest_img_rec.growth_features or {
                    'exg_greenness': latest_img_rec.actual_greenness_score or 55.0,
                    'foliage_density': 0.72,
                    'aspect_ratio': 1.10,
                    'leaf_clusters': 5,
                    'canopy_spread_width_cm': 18.0,
                },
                'image_features_used': True,
            }

        result = predict_growth(
            ph          = sensor.ph           if sensor.ph is not None else 6.6,
            nitrogen    = sensor.npk_nitrogen  if sensor.npk_nitrogen is not None else 60.0,
            phosphorus  = sensor.npk_phosphorus if sensor.npk_phosphorus is not None else 32.0,
            potassium   = sensor.npk_potassium  if sensor.npk_potassium is not None else 52.0,
            temperature = sensor.temperature   if sensor.temperature is not None else 28.0,
            humidity    = sensor.humidity      if sensor.humidity is not None else 65.0,
            moisture    = sensor.soil_moisture  if sensor.soil_moisture is not None else 48.0,
            day_number  = session.days_since_sowing,
            field_acres = session.field_acres,
            session     = session,
            image_features = img_feats,
        )

        GrowthPrediction.objects.create(
            sensor_data              = sensor,
            predicted_yield_per_acre = result['predicted_yield_per_acre'],
            growth_stage             = result['growth_stage'],
            health_score             = result['health_score'],
            recommendation           = result['recommendation_text'],
            model_accuracy           = result['model_accuracy'],
        )

        today = date.today()
        slot  = 'morning' if timezone.now().hour < 14 else 'evening'
        record, created = DailyGrowthRecord.objects.get_or_create(
            session=session, date=today, slot=slot,
            defaults={
                'day_number':               session.days_since_sowing,
                'predicted_height_cm':      result['predicted_height_cm'],
                'predicted_yield_per_acre': result['predicted_yield_per_acre'],
                'health_score':             result['health_score'],
                'growth_stage':             result['growth_stage'],
                'recommendation':           result['recommendation_text'],
                'model_accuracy':           result['model_accuracy'],
                'sensor_data':              sensor,
                'multimodal_state':         result['multimodal_state'],
                'growth_trend_data':        result['growth_trend'],
                'risk_anomaly_data':        result['risk_anomaly'],
                'prediction_data':          result['prediction'],
                'irrigation_suggestion':    result['irrigation_suggestion'],
                'nutrient_suggestion':      result['nutrient_suggestion'],
                'crop_health_warning':      result['crop_health_warning'],
            }
        )
        if not created and not record.image_uploaded:
            record.predicted_height_cm      = result['predicted_height_cm']
            record.predicted_yield_per_acre = result['predicted_yield_per_acre']
            record.health_score             = result['health_score']
            record.growth_stage             = result['growth_stage']
            record.recommendation           = result['recommendation_text']
            record.model_accuracy           = result['model_accuracy']
            record.sensor_data              = sensor
            record.multimodal_state         = result['multimodal_state']
            record.growth_trend_data        = result['growth_trend']
            record.risk_anomaly_data        = result['risk_anomaly']
            record.prediction_data          = result['prediction']
            record.irrigation_suggestion    = result['irrigation_suggestion']
            record.nutrient_suggestion      = result['nutrient_suggestion']
            record.crop_health_warning      = result['crop_health_warning']
            record.save()

        maybe_trigger_retrain()

        return JsonResponse({
            'status':                   'ok',
            'sensor_id':                sensor.id,
            'predicted_yield_per_acre': result['predicted_yield_per_acre'],
            'predicted_height_cm':      result['predicted_height_cm'],
            'health_score':             result['health_score'],
            'growth_stage':             result['growth_stage'],
            'model_accuracy':           result['model_accuracy'],
            'model_version':            result.get('model_version', '3.0-gradient-boosting-xgboost'),
            'image_features_used':      result.get('image_features_used', False),
            'multimodal_state':         result['multimodal_state'],
            'growth_trend':             result['growth_trend'],
            'risk_anomaly':             result['risk_anomaly'],
            'prediction':               result['prediction'],
            'irrigation_suggestion':    result['irrigation_suggestion'],
            'nutrient_suggestion':      result['nutrient_suggestion'],
            'crop_health_warning':      result['crop_health_warning'],
            'recommendations':          result['recommendations'],
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@require_http_methods(["GET"])
def api_live_data(request, session_id):
    """Live sensor + latest multimodal prediction polled every 5 s."""
    session = get_object_or_404(CropSession, id=session_id)
    sensor  = session.sensor_readings.order_by('-timestamp').first()
    if not sensor:
        return JsonResponse({'status': 'no_data'})

    pred = sensor.predictions.order_by('-timestamp').first()
    history = list(
        session.sensor_readings
        .values('timestamp', 'ph', 'temperature', 'humidity',
                'soil_moisture', 'npk_nitrogen', 'npk_phosphorus', 'npk_potassium')
        [:20]
    )
    for h in history:
        h['timestamp'] = h['timestamp'].strftime('%Y-%m-%d %H:%M')

    latest_rec = (
        session.growth_records
        .exclude(health_score__isnull=True)
        .order_by('-date', '-slot')
        .first()
    )

    return JsonResponse({
        'status': 'ok',
        'latest_sensor': {
            'ph':            sensor.ph,
            'temperature':   sensor.temperature,
            'humidity':      sensor.humidity,
            'soil_moisture': sensor.soil_moisture,
            'npk_n':         sensor.npk_nitrogen,
            'npk_p':         sensor.npk_phosphorus,
            'npk_k':         sensor.npk_potassium,
            'device_id':     sensor.device_id,
            'timestamp':     sensor.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'latest_prediction': {
            'predicted_yield_per_acre': pred.predicted_yield_per_acre if pred else None,
            'health_score':             pred.health_score  if pred else None,
            'growth_stage':             pred.growth_stage  if pred else None,
            'model_accuracy':           pred.model_accuracy if pred else None,
        } if pred else None,
        'growth_trend':          latest_rec.growth_trend_data if latest_rec and latest_rec.growth_trend_data else None,
        'risk_anomaly':          latest_rec.risk_anomaly_data if latest_rec and latest_rec.risk_anomaly_data else None,
        'irrigation_suggestion': latest_rec.irrigation_suggestion if latest_rec and latest_rec.irrigation_suggestion else None,
        'nutrient_suggestion':   latest_rec.nutrient_suggestion if latest_rec and latest_rec.nutrient_suggestion else None,
        'crop_health_warning':   latest_rec.crop_health_warning if latest_rec and latest_rec.crop_health_warning else None,
        'latest_image_metrics': {
            'greenness_score':       latest_rec.actual_greenness_score if latest_rec else None,
            'leaf_coverage_percent': latest_rec.actual_leaf_coverage_percent if latest_rec else None,
            'leaf_area_cm2':         latest_rec.actual_leaf_area_cm2 if latest_rec else None,
            'canopy_size_cm2':       latest_rec.canopy_size_cm2 if latest_rec else None,
            'actual_height_cm':      latest_rec.actual_height_cm if latest_rec else None,
        } if latest_rec else None,
        'days_since_sowing': session.days_since_sowing,
        'history': list(reversed(history)),
    })


@require_http_methods(["GET"])
def api_predict_growth(request, session_id):
    """On-demand growth prediction with optional sensor query params."""
    session = get_object_or_404(CropSession, id=session_id)
    try:
        latest = session.sensor_readings.order_by('-timestamp').first()

        # Pull latest image features from the most recent record that has them
        latest_img_rec = (
            session.growth_records
            .exclude(actual_greenness_score__isnull=True)
            .order_by('-date', '-slot')
            .first()
        )

        result = predict_growth(
            ph          = float(request.GET.get('ph',       latest.ph            if latest else 6.5)),
            nitrogen    = float(request.GET.get('n',        latest.npk_nitrogen   if latest else 60)),
            phosphorus  = float(request.GET.get('p',        latest.npk_phosphorus if latest else 30)),
            potassium   = float(request.GET.get('k',        latest.npk_potassium  if latest else 50)),
            temperature = float(request.GET.get('temp',     latest.temperature    if latest else 27)),
            humidity    = float(request.GET.get('humidity', latest.humidity       if latest else 65)),
            moisture    = float(request.GET.get('moisture', latest.soil_moisture  if latest else 45)),
            day_number  = session.days_since_sowing,
            field_acres = session.field_acres,
            # Use cached image features from the latest growth record that has them
            greenness_score       = latest_img_rec.actual_greenness_score       if latest_img_rec else None,
            leaf_coverage_percent = latest_img_rec.actual_leaf_coverage_percent if latest_img_rec else None,
            leaf_area_norm        = (latest_img_rec.actual_leaf_coverage_percent / 100.0)
                                    if latest_img_rec and latest_img_rec.actual_leaf_coverage_percent else None,
        )
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@require_http_methods(["GET"])
def api_growth_chart(request, session_id):
    """
    Chart data: Structured explicitly for the 4 DASHBOARD GRAPHS:
      1. Plant Height Trend (Observed vs Predicted vs Ideal)
      2. Canopy Growth (Canopy Coverage % & Canopy Size cm² & Expansion Velocity)
      3. Sensor Graphs (Continuous readings: Temp, Humidity, Moisture, pH, NPK)
      4. Predicted Yield (Projected Harvest Forecast kg/acre vs Target)
    """
    session = get_object_or_404(CropSession, id=session_id)
    records = session.growth_records.order_by('day_number', 'slot')

    seen = {}
    days                    = []
    actual_heights          = []
    predicted_heights       = []
    predicted_yields        = []
    health_trend            = []
    greenness_actual        = []
    leaf_coverage_trend     = []
    canopy_size_trend       = []
    expansion_velocities    = []
    prev_cov                = None

    for r in records:
        if r.day_number in seen:
            continue
        seen[r.day_number] = True
        days.append(r.day_number)
        actual_heights.append(r.actual_height_cm)
        predicted_heights.append(r.predicted_height_cm or r.actual_height_cm)
        predicted_yields.append(r.predicted_yield_per_acre)
        health_trend.append(r.health_score)
        greenness_actual.append(r.actual_greenness_score)

        cov = r.actual_leaf_coverage_percent or 35.0
        leaf_coverage_trend.append(cov)
        canopy_size_trend.append(r.canopy_size_cm2 or round(cov * 9.0, 1))

        if prev_cov is not None:
            expansion_velocities.append(round(cov - prev_cov, 1))
        else:
            expansion_velocities.append(round(cov * 0.1, 1))
        prev_cov = cov

    total_days = 60
    if session.expected_harvest:
        total_days = max(30, (session.expected_harvest - session.sowing_date).days)
    ideal_days, ideal_heights, ideal_yields = _ideal_curve(total_days)

    # Continuous sensor readings for Sensor Graphs
    sensor_readings = list(
        session.sensor_readings
        .values('timestamp', 'temperature', 'humidity', 'soil_moisture', 'ph', 'npk_nitrogen', 'npk_phosphorus', 'npk_potassium')
        .order_by('-timestamp')[:30]
    )
    sensor_history_series = list(reversed(sensor_readings))
    for s in sensor_history_series:
        s['time_label'] = s['timestamp'].strftime('%d %b %H:%M')

    return JsonResponse({
        # 1. Plant Height Trend Graph
        'plant_height_trend': {
            'days':              days,
            'actual_heights':    actual_heights,
            'predicted_heights': predicted_heights,
            'ideal_heights':     ideal_heights[:len(days)] if len(ideal_heights) >= len(days) else ideal_heights,
        },
        # 2. Canopy Growth Graph
        'canopy_growth': {
            'days':                 days,
            'leaf_coverage_percent': leaf_coverage_trend,
            'canopy_size_cm2':      canopy_size_trend,
            'expansion_velocities': expansion_velocities,
        },
        # 3. Sensor Graphs (Continuous multi-parameter readings)
        'sensor_graphs': {
            'timestamps':    [s['time_label'] for s in sensor_history_series],
            'temperature':   [s['temperature'] for s in sensor_history_series],
            'humidity':      [s['humidity'] for s in sensor_history_series],
            'soil_moisture': [s['soil_moisture'] for s in sensor_history_series],
            'ph':            [s['ph'] for s in sensor_history_series],
            'nitrogen':      [s['npk_nitrogen'] for s in sensor_history_series],
            'phosphorus':    [s['npk_phosphorus'] for s in sensor_history_series],
            'potassium':     [s['npk_potassium'] for s in sensor_history_series],
        },
        # 4. Predicted Yield Graph
        'predicted_yield': {
            'days':             days,
            'predicted_yields': predicted_yields,
            'ideal_yields':     ideal_yields[:len(days)] if len(ideal_yields) >= len(days) else ideal_yields,
            'target_harvest_yield': 650.0,
        },
        # Legacy chart keys for backwards-compatibility
        'days':                  days,
        'actual_heights':        actual_heights,
        'predicted_heights':     predicted_heights,
        'predicted_yields':      predicted_yields,
        'health_trend':          health_trend,
        'greenness_actual':      greenness_actual,
        'leaf_coverage_trend':   leaf_coverage_trend,
        'ideal_days':            ideal_days,
        'ideal_heights':         ideal_heights,
        'ideal_yields':          ideal_yields,
        'crop':                  session.crop_name,
        'total_days':            total_days,
    })


@csrf_exempt
@require_http_methods(["POST"])
def api_upload_growth_image(request, session_id):
    """
    DAILY IMAGE (Morning / Evening Capture):
    Feeds image into IMAGE GROWTH MODEL:
      - Plant area
      - Green coverage
      - Canopy size
      - Plant height (estimated)
      - Growth features
    Merges with continuous Sensor Data into MULTIMODAL GROWTH STATE.
    Executes GROWTH PREDICTION MODEL (Gradient Boosting / XGBoost).
    Generates RECOMMENDATION ENGINE outputs.
    Updates DASHBOARD GRAPHS data.
    """
    session = get_object_or_404(CropSession, id=session_id)
    try:
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image provided'}, status=400)

        slot = request.POST.get('slot', 'morning' if timezone.now().hour < 14 else 'evening')

        plant_img = PlantImage.objects.create(
            image       = image_file,
            farmer_name = session.farmer_name,
            location    = session.location,
            notes       = request.POST.get('notes', ''),
        )

        # ── 1. IMAGE GROWTH MODEL ──────────────────────────────────────────
        img_feats = extract_image_growth_model(
            plant_img.image.path,
            frame_area_cm2=session.camera_frame_area_cm2,
            day_number=session.days_since_sowing,
        )

        plant_area_px          = img_feats['plant_area_px']
        plant_area_cm2         = img_feats['plant_area_cm2']
        green_coverage_percent = img_feats['green_coverage_percent']
        canopy_size_cm2        = img_feats['canopy_size_cm2']
        estimated_plant_height = img_feats['estimated_plant_height_cm']
        growth_features        = img_feats['growth_features']
        greenness              = growth_features['exg_greenness']

        # ── Disease screening (ResNet-18) ──────────────────────────────────
        disease_res = predict_disease(plant_img.image.path)
        diseased, disease_name, confidence, model_acc, treatment, low_confidence = disease_res[:6]

        # ── 2. MULTIMODAL STATE FUSION & 3. PREDICTION MODEL ───────────────
        latest_sensor = session.sensor_readings.order_by('-timestamp').first()
        growth_result = predict_growth(
            ph          = latest_sensor.ph            if latest_sensor and latest_sensor.ph is not None else 6.6,
            nitrogen    = latest_sensor.npk_nitrogen   if latest_sensor and latest_sensor.npk_nitrogen is not None else 60.0,
            phosphorus  = latest_sensor.npk_phosphorus if latest_sensor and latest_sensor.npk_phosphorus is not None else 32.0,
            potassium   = latest_sensor.npk_potassium  if latest_sensor and latest_sensor.npk_potassium is not None else 52.0,
            temperature = latest_sensor.temperature    if latest_sensor and latest_sensor.temperature is not None else 28.0,
            humidity    = latest_sensor.humidity       if latest_sensor and latest_sensor.humidity is not None else 65.0,
            moisture    = latest_sensor.soil_moisture  if latest_sensor and latest_sensor.soil_moisture is not None else 48.0,
            day_number  = session.days_since_sowing,
            field_acres = session.field_acres,
            session     = session,
            image_features = img_feats,
        )

        # ── Save to PlantImage ──────────────────────────────────────────────
        plant_img.disease_detected      = diseased
        plant_img.disease_name          = disease_name
        plant_img.confidence            = confidence
        plant_img.model_accuracy        = model_acc
        plant_img.low_confidence        = low_confidence
        plant_img.processed             = True
        plant_img.greenness_score       = greenness
        plant_img.leaf_coverage_percent = green_coverage_percent
        plant_img.leaf_area_cm2         = plant_area_cm2
        plant_img.save()

        # ── Upsert DailyGrowthRecord ────────────────────────────────────────
        today  = date.today()
        record, _ = DailyGrowthRecord.objects.get_or_create(
            session=session, date=today, slot=slot,
            defaults={'day_number': session.days_since_sowing},
        )
        record.image                        = plant_img
        record.image_uploaded               = True
        record.actual_greenness_score       = greenness
        record.actual_leaf_coverage_percent = green_coverage_percent
        record.actual_leaf_area_cm2         = plant_area_cm2
        record.canopy_size_cm2              = canopy_size_cm2
        record.estimated_plant_height_cm    = estimated_plant_height
        record.actual_height_cm             = estimated_plant_height
        record.growth_features              = growth_features

        record.predicted_height_cm          = growth_result['predicted_height_cm']
        record.predicted_yield_per_acre     = growth_result['predicted_yield_per_acre']
        record.health_score                 = growth_result['health_score']
        record.growth_stage                 = growth_result['growth_stage']
        record.model_accuracy               = growth_result['model_accuracy']
        record.sensor_data                  = latest_sensor
        record.multimodal_state             = growth_result['multimodal_state']
        record.growth_trend_data            = growth_result['growth_trend']
        record.risk_anomaly_data            = growth_result['risk_anomaly']
        record.prediction_data              = growth_result['prediction']
        record.irrigation_suggestion        = growth_result['irrigation_suggestion']
        record.nutrient_suggestion          = growth_result['nutrient_suggestion']
        record.crop_health_warning          = growth_result['crop_health_warning']

        rec_text = growth_result['recommendation_text']
        if diseased:
            rec_text = (rec_text + ' | ' if rec_text else '') + f'Disease detected: {disease_name}'
        record.recommendation = rec_text
        record.save()

        maybe_trigger_retrain()

        if diseased:
            CropNote.objects.create(
                session  = session,
                category = 'alert',
                title    = f'Disease detected: {disease_name}',
                body     = f'Confidence: {confidence}%. {treatment.get("additional", "")}',
                image    = plant_img,
            )

        return JsonResponse({
            'status':                   'ok',
            'image_id':                 plant_img.id,
            'image_url':                plant_img.image.url,
            'slot':                     slot,
            'day_number':               record.day_number,
            # Image Growth Model outputs
            'plant_area_px':            plant_area_px,
            'plant_area_cm2':           plant_area_cm2,
            'green_coverage_percent':   green_coverage_percent,
            'canopy_size_cm2':          canopy_size_cm2,
            'estimated_plant_height_cm': estimated_plant_height,
            'growth_features':          growth_features,
            # Disease screening
            'disease_detected':         diseased,
            'disease_name':             disease_name,
            'confidence':               confidence,
            'model_accuracy':           model_acc,
            'low_confidence':           low_confidence,
            'treatment':                treatment,
            # Multimodal outputs
            'multimodal_state':         growth_result['multimodal_state'],
            'growth_trend':             growth_result['growth_trend'],
            'risk_anomaly':             growth_result['risk_anomaly'],
            'prediction':               growth_result['prediction'],
            'irrigation_suggestion':    growth_result['irrigation_suggestion'],
            'nutrient_suggestion':      growth_result['nutrient_suggestion'],
            'crop_health_warning':      growth_result['crop_health_warning'],
            'predicted_yield_per_acre': growth_result['predicted_yield_per_acre'],
            'health_score':             growth_result['health_score'],
            'growth_stage':             growth_result['growth_stage'],
            'recommendations':          growth_result['recommendations'],
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def api_multimodal_pipeline(request, session_id):
    """
    Returns the live data flowing through every node of the Growth Monitoring Architecture:
      [Same Crop Session] -> [Daily Image + Sensor Data] -> [Image Growth Model + Sensor Processing]
      -> [Multimodal Growth State] -> [Growth Prediction Model] -> [Growth Trend, Risk/Anomaly, Prediction]
      -> [Recommendation Engine] -> [Dashboard Graphs] -> [Loopback to Session]
    """
    try:
        session = get_object_or_404(CropSession, id=session_id)
        latest_sensor = session.sensor_readings.order_by('-timestamp').first()
        latest_rec    = (
            session.growth_records
            .exclude(health_score__isnull=True)
            .order_by('-date', '-slot')
            .first()
        )

        latest_img_rec = (
            session.growth_records
            .exclude(actual_greenness_score__isnull=True)
            .order_by('-date', '-slot')
            .first()
        )

        img_feats = None
        if latest_img_rec:
            img_feats = {
                'plant_area_px': 25000,
                'plant_area_cm2': latest_img_rec.actual_leaf_area_cm2 or 300.0,
                'green_coverage_percent': latest_img_rec.actual_leaf_coverage_percent or 40.0,
                'canopy_size_cm2': latest_img_rec.canopy_size_cm2 or 360.0,
                'estimated_plant_height_cm': latest_img_rec.estimated_plant_height_cm or 24.0,
                'growth_features': latest_img_rec.growth_features or {
                    'exg_greenness': latest_img_rec.actual_greenness_score or 55.0,
                    'foliage_density': 0.72,
                    'aspect_ratio': 1.10,
                    'leaf_clusters': 5,
                    'canopy_spread_width_cm': 18.0,
                },
                'image_features_used': True,
            }

        res = predict_growth(
            ph          = latest_sensor.ph            if latest_sensor and latest_sensor.ph is not None else 6.6,
            nitrogen    = latest_sensor.npk_nitrogen   if latest_sensor and latest_sensor.npk_nitrogen is not None else 60.0,
            phosphorus  = latest_sensor.npk_phosphorus if latest_sensor and latest_sensor.npk_phosphorus is not None else 32.0,
            potassium   = latest_sensor.npk_potassium  if latest_sensor and latest_sensor.npk_potassium is not None else 52.0,
            temperature = latest_sensor.temperature    if latest_sensor and latest_sensor.temperature is not None else 28.0,
            humidity    = latest_sensor.humidity       if latest_sensor and latest_sensor.humidity is not None else 65.0,
            moisture    = latest_sensor.soil_moisture  if latest_sensor and latest_sensor.soil_moisture is not None else 48.0,
            day_number  = session.days_since_sowing,
            field_acres = session.field_acres,
            session     = session,
            image_features = img_feats,
        )

        today = date.today()
        morning_img = session.growth_records.filter(date=today, slot='morning', image_uploaded=True).first()
        evening_img = session.growth_records.filter(date=today, slot='evening', image_uploaded=True).first()

        return JsonResponse({
            'status': 'ok',
            'session': {
                'id': session.id,
                'crop_name': session.crop_name,
                'variety': session.crop_variety,
                'days_since_sowing': session.days_since_sowing,
                'status': session.status,
                'progress_pct': session.progress_pct,
                'is_continuous': True,
            },
            'daily_image': {
                'morning_capture': {
                    'completed': bool(morning_img),
                    'image_url': morning_img.image.image.url if morning_img and morning_img.image else None,
                    'time_slot': '06:00 - 12:00',
                },
                'evening_capture': {
                    'completed': bool(evening_img),
                    'image_url': evening_img.image.image.url if evening_img and evening_img.image else None,
                    'time_slot': '15:00 - 19:00',
                },
            },
            'image_growth_model': res['multimodal_state']['image_stream'],
            'sensor_data': {
                'temperature': latest_sensor.temperature if latest_sensor else 28.0,
                'humidity': latest_sensor.humidity if latest_sensor else 65.0,
                'soil_moisture': latest_sensor.soil_moisture if latest_sensor else 48.0,
                'ph': latest_sensor.ph if latest_sensor else 6.6,
                'npk': {
                    'n': latest_sensor.npk_nitrogen if latest_sensor else 60.0,
                    'p': latest_sensor.npk_phosphorus if latest_sensor else 32.0,
                    'k': latest_sensor.npk_potassium if latest_sensor else 52.0,
                },
            },
            'sensor_processing': res['multimodal_state']['sensor_stream'],
            'multimodal_growth_state': res['multimodal_state'],
            'growth_prediction_model': {
                'architecture': 'Gradient Boosting / XGBoost',
                'growth_trend': res['growth_trend'],
                'risk_anomaly': res['risk_anomaly'],
                'prediction': res['prediction'],
            },
            'recommendation_engine': {
                'irrigation_suggestion': res['irrigation_suggestion'],
                'nutrient_suggestion': res['nutrient_suggestion'],
                'crop_health_warning': res['crop_health_warning'],
            },
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def api_notes(request, session_id):
    session = get_object_or_404(CropSession, id=session_id)
    notes   = session.crop_notes.order_by('-created_at')[:30]
    return JsonResponse({'notes': [{
        'id':         n.id,
        'category':   n.category,
        'title':      n.title,
        'body':       n.body,
        'created_at': n.created_at.strftime('%d %b %Y, %H:%M'),
        'image_url':  n.image.image.url if n.image else None,
    } for n in notes]})


@csrf_exempt
@require_http_methods(["POST"])
def api_add_note(request, session_id):
    session = get_object_or_404(CropSession, id=session_id)
    try:
        body = json.loads(request.body)
        note = CropNote.objects.create(
            session  = session,
            category = body.get('category', 'observation'),
            title    = body.get('title', '').strip(),
            body     = body.get('body', '').strip(),
        )
        return JsonResponse({
            'status': 'ok',
            'note': {
                'id':         note.id,
                'category':   note.category,
                'title':      note.title,
                'body':       note.body,
                'created_at': note.created_at.strftime('%d %b %Y, %H:%M'),
            }
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


# ═══════════════════════════════════════════════════════════════════════════
# DISEASE DETECTION APIs (standalone)
# ═══════════════════════════════════════════════════════════════════════════

@csrf_exempt
@require_http_methods(["POST"])
def api_disease_analyse(request):
    try:
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image provided'}, status=400)

        plant_img = PlantImage.objects.create(
            image       = image_file,
            farmer_name = request.POST.get('farmer_name', ''),
            location    = request.POST.get('location', ''),
            notes       = request.POST.get('notes', ''),
        )

        disease_res = predict_disease(plant_img.image.path)
        diseased, disease_name, confidence, model_acc, treatment, low_confidence = disease_res[:6]
        top_predictions = disease_res[6] if len(disease_res) > 6 else []

        img_feats = extract_image_features(plant_img.image.path)
        greenness             = img_feats['greenness_score']
        leaf_coverage_percent = img_feats['leaf_coverage_percent']
        leaf_area_cm2         = img_feats['leaf_area_cm2']

        plant_img.disease_detected      = diseased
        plant_img.disease_name          = disease_name
        plant_img.confidence            = confidence
        plant_img.model_accuracy        = model_acc
        plant_img.low_confidence        = low_confidence
        plant_img.processed             = True
        plant_img.greenness_score       = greenness
        plant_img.leaf_coverage_percent = leaf_coverage_percent
        plant_img.leaf_area_cm2         = leaf_area_cm2
        plant_img.save()

        return JsonResponse({
            'status':                'ok',
            'image_id':              plant_img.id,
            'image_url':             plant_img.image.url,
            'disease_detected':      diseased,
            'disease_name':          disease_name,
            'formatted_name':        format_disease_name(disease_name),
            'confidence':            confidence,
            'model_accuracy':        model_acc,
            'low_confidence':        low_confidence,
            'greenness_score':       greenness,
            'leaf_coverage_percent': leaf_coverage_percent,
            'leaf_area_cm2':         leaf_area_cm2,
            'treatment':             treatment,
            'top_predictions':       top_predictions,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def api_disease_sample(request):
    """Run disease detection against one of the pre-loaded static sample leaves."""
    try:
        from pathlib import Path
        body = json.loads(request.body)
        sample_key = body.get('sample', 'healthy_leaf')
        sample_file_map = {
            'healthy_leaf': 'healthy_leaf.jpg',
            'early_blight': 'early_blight.jpg',
            'rust_leaf': 'rust_leaf.jpg',
            'black_measles': 'black_measles.jpg',
        }
        filename = sample_file_map.get(sample_key, 'healthy_leaf.jpg')
        sample_path = Path(settings.STATICFILES_DIRS[0]) / 'samples' / filename
        if not sample_path.exists():
            return JsonResponse({'error': 'Sample file not found'}, status=404)

        from django.core.files import File
        plant_img = PlantImage.objects.create(
            farmer_name='Demo Sample',
            location='Pathology Lab Reference',
            notes=f'Analysis of {sample_key} reference specimen',
        )
        with open(sample_path, 'rb') as f:
            plant_img.image.save(filename, File(f), save=True)

        disease_res = predict_disease(plant_img.image.path)
        diseased, disease_name, confidence, model_acc, treatment, low_confidence = disease_res[:6]
        top_predictions = disease_res[6] if len(disease_res) > 6 else []

        img_feats = extract_image_features(plant_img.image.path)
        greenness = img_feats['greenness_score']
        leaf_coverage_percent = img_feats['leaf_coverage_percent']
        leaf_area_cm2 = img_feats['leaf_area_cm2']

        plant_img.disease_detected = diseased
        plant_img.disease_name = disease_name
        plant_img.confidence = confidence
        plant_img.model_accuracy = model_acc
        plant_img.low_confidence = low_confidence
        plant_img.processed = True
        plant_img.greenness_score = greenness
        plant_img.leaf_coverage_percent = leaf_coverage_percent
        plant_img.leaf_area_cm2 = leaf_area_cm2
        plant_img.save()

        return JsonResponse({
            'status': 'ok',
            'image_id': plant_img.id,
            'image_url': plant_img.image.url,
            'disease_detected': diseased,
            'disease_name': disease_name,
            'formatted_name': format_disease_name(disease_name),
            'confidence': confidence,
            'model_accuracy': model_acc,
            'low_confidence': low_confidence,
            'greenness_score': greenness,
            'leaf_coverage_percent': leaf_coverage_percent,
            'leaf_area_cm2': leaf_area_cm2,
            'treatment': treatment,
            'top_predictions': top_predictions,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def api_disease_history(request):
    page     = int(request.GET.get('page', 1))
    per_page = 20
    qs       = PlantImage.objects.filter(processed=True).order_by('-uploaded_at')
    total    = qs.count()
    images   = qs[(page - 1) * per_page: page * per_page]
    return JsonResponse({
        'total': total,
        'page':  page,
        'images': [{
            'id':                    img.id,
            'url':                   img.image.url,
            'disease_detected':      img.disease_detected,
            'disease_name':          format_disease_name(img.disease_name) if img.disease_name else 'Unprocessed',
            'raw_disease_name':      img.disease_name,
            'confidence':            img.confidence,
            'model_accuracy':        img.model_accuracy,
            'low_confidence':        img.low_confidence,
            'greenness_score':       img.greenness_score,
            'leaf_coverage_percent': img.leaf_coverage_percent,
            'leaf_area_cm2':         img.leaf_area_cm2,
            'farmer_name':           img.farmer_name,
            'location':              img.location,
            'uploaded_at':           img.uploaded_at.strftime('%d %b %Y, %H:%M'),
        } for img in images],
    })


# ═══════════════════════════════════════════════════════════════════════════
# LEGACY (keeps existing ESP32 firmware working with old URL)
# ═══════════════════════════════════════════════════════════════════════════

@csrf_exempt
@require_http_methods(["POST"])
def api_sensor_data_legacy(request):
    active = CropSession.objects.filter(status='active').order_by('-created_at').first()
    if not active:
        active = CropSession.objects.create(
            crop_name='Green Gram', field_acres=1.0, sowing_date=date.today()
        )
    return api_sensor_data(request, session_id=active.id)
