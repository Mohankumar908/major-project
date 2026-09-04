import json
import math
from datetime import date
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.db.models import Avg, Count, Max

from .models import (
    SensorData, GrowthPrediction, PlantImage,
    CropSession, DailyGrowthRecord, CropNote,
)
from .ml_utils import predict_growth, predict_disease, extract_greenness


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

CROP_ICONS = {
    'Green Gram': '🫘', 'Rice': '🌾', 'Wheat': '🌿', 'Tomato': '🍅',
    'Maize': '🌽',      'Potato': '🥔', 'Soybean': '🫘', 'Sugarcane': '🎋',
    'Cotton': '☁️',     'Other': '🌱',
}

def _session_summary(session):
    """Return a serialisable dict for use in home page cards and the sessions API."""
    latest_rec = (
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
            'ph':          latest_sensor.ph,
            'temperature': latest_sensor.temperature,
            'humidity':    latest_sensor.humidity,
            'soil_moisture': latest_sensor.soil_moisture,
            'timestamp':   latest_sensor.timestamp.strftime('%Y-%m-%d %H:%M'),
        } if latest_sensor else None,
    }


def _ideal_curve(total_days=60):
    """Sigmoid height + ramp yield ideal baseline."""
    ideal_days, ideal_heights, ideal_yields = [], [], []
    for d in range(1, total_days + 1):
        h = round(5 + 45 * (1 / (1 + math.exp(-0.12 * (d - 28)))), 1)
        y = round(200 + 350 * (d / total_days) ** 1.5, 1)
        ideal_days.append(d)
        ideal_heights.append(h)
        ideal_yields.append(y)
    return ideal_days, ideal_heights, ideal_yields


# ═══════════════════════════════════════════════════════════════════════════
# PAGE VIEWS
# ═══════════════════════════════════════════════════════════════════════════

def home(request):
    """Session list — the entry point of the app."""
    active_sessions    = CropSession.objects.filter(status='active').order_by('-created_at')
    completed_sessions = CropSession.objects.filter(status='completed').order_by('-created_at')
    paused_sessions    = CropSession.objects.filter(status='paused').order_by('-created_at')

    # Stats for the top summary bar
    total_active    = active_sessions.count()
    total_completed = completed_sessions.count()
    disease_alerts  = PlantImage.objects.filter(
        processed=True, disease_detected=True
    ).count()

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
    """Per-crop monitoring page."""
    session = get_object_or_404(CropSession, id=session_id)

    # Latest sensor & prediction
    latest_sensor = session.sensor_readings.order_by('-timestamp').first()
    latest_rec    = (
        session.growth_records
        .exclude(health_score__isnull=True)
        .order_by('-date', '-slot')
        .first()
    )

    # Chart data — per-day, pick latest slot
    growth_records = (
        session.growth_records
        .order_by('day_number', 'slot')[:90]
    )
    chart_days, actual_heights, pred_heights, pred_yields = [], [], [], []
    seen = {}
    for r in growth_records:
        if r.day_number not in seen:
            seen[r.day_number] = True
            chart_days.append(r.day_number)
            actual_heights.append(r.actual_height_cm)
            pred_heights.append(r.predicted_height_cm)
            pred_yields.append(r.predicted_yield_per_acre)

    # Sensor history for sparkline (last 20)
    sensor_history = list(
        session.sensor_readings
        .values('timestamp', 'ph', 'temperature', 'humidity', 'soil_moisture', 'npk_nitrogen')
        [:20]
    )
    for s in sensor_history:
        s['timestamp'] = s['timestamp'].strftime('%H:%M')

    # Today's upload status
    today        = date.today()
    morning_done = session.growth_records.filter(date=today, slot='morning', image_uploaded=True).exists()
    evening_done = session.growth_records.filter(date=today, slot='evening', image_uploaded=True).exists()

    # Recent notes (latest 5)
    recent_notes = session.crop_notes.order_by('-created_at')[:5]

    # Recent growth images linked to this session
    recent_images = (
        PlantImage.objects.filter(
            dailygrowthrecord__session=session,
            processed=True,
        )
        .distinct()
        .order_by('-uploaded_at')[:8]
    )

    # Avg health over last 7 records
    avg_health = (
        session.growth_records
        .exclude(health_score__isnull=True)
        .order_by('-date')[:7]
        .aggregate(avg=Avg('health_score'))['avg']
    )

    return render(request, 'agriculture/session_detail.html', {
        'session':              session,
        'latest_sensor':        latest_sensor,
        'latest_rec':           latest_rec,
        'morning_done':         morning_done,
        'evening_done':         evening_done,
        'recent_notes':         recent_notes,
        'recent_images':        recent_images,
        'avg_health':           round(avg_health, 1) if avg_health else None,
        'icon':                 CROP_ICONS.get(session.crop_name, '🌱'),
        # JSON for charts
        'sensor_history_json':  json.dumps(list(reversed(sensor_history))),
        'chart_days_json':      json.dumps(chart_days),
        'actual_heights_json':  json.dumps(actual_heights),
        'pred_heights_json':    json.dumps(pred_heights),
        'pred_yields_json':     json.dumps(pred_yields),
    })


def disease_detection(request):
    """Standalone disease detection page — no session required."""
    recent = PlantImage.objects.filter(processed=True).order_by('-uploaded_at')[:20]
    total_scans    = PlantImage.objects.filter(processed=True).count()
    disease_found  = PlantImage.objects.filter(processed=True, disease_detected=True).count()
    healthy_found  = PlantImage.objects.filter(processed=True, disease_detected=False).count()
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
        body = json.loads(request.body)
        sowing = body.get('sowing_date') or date.today().isoformat()
        session = CropSession.objects.create(
            crop_name     = body.get('crop_name', 'Green Gram'),
            crop_variety  = body.get('crop_variety', ''),
            field_name    = body.get('field_name', ''),
            field_acres   = float(body.get('field_acres', 1.0)),
            farmer_name   = body.get('farmer_name', ''),
            location      = body.get('location', ''),
            sowing_date   = sowing,
            expected_harvest = body.get('expected_harvest') or None,
            notes         = body.get('notes', ''),
            status        = 'active',
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


# ═══════════════════════════════════════════════════════════════════════════
# GROWTH MONITORING APIs  (all scoped to a session)
# ═══════════════════════════════════════════════════════════════════════════

@csrf_exempt
@require_http_methods(["POST"])
def api_sensor_data(request, session_id):
    """Receive IoT sensor payload and run growth prediction."""
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
        result = predict_growth(
            sensor.ph or 6.5, sensor.npk_nitrogen or 60, sensor.npk_phosphorus or 30,
            sensor.npk_potassium or 50, sensor.temperature or 27,
            sensor.humidity or 65, sensor.soil_moisture or 45,
            day_number=session.days_since_sowing, field_acres=session.field_acres,
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
            record.save()

        return JsonResponse({
            'status':                 'ok',
            'sensor_id':              sensor.id,
            'predicted_yield_per_acre': result['predicted_yield_per_acre'],
            'predicted_height_cm':    result['predicted_height_cm'],
            'health_score':           result['health_score'],
            'growth_stage':           result['growth_stage'],
            'model_accuracy':         result['model_accuracy'],
            'recommendations':        result['recommendations'],
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@require_http_methods(["GET"])
def api_live_data(request, session_id):
    """Live sensor + latest prediction for a session (polled every 5 s)."""
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

    return JsonResponse({
        'status': 'ok',
        'latest_sensor': {
            'ph':           sensor.ph,
            'temperature':  sensor.temperature,
            'humidity':     sensor.humidity,
            'soil_moisture': sensor.soil_moisture,
            'npk_n':        sensor.npk_nitrogen,
            'npk_p':        sensor.npk_phosphorus,
            'npk_k':        sensor.npk_potassium,
            'device_id':    sensor.device_id,
            'timestamp':    sensor.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'latest_prediction': {
            'predicted_yield_per_acre': pred.predicted_yield_per_acre if pred else None,
            'health_score':  pred.health_score  if pred else None,
            'growth_stage':  pred.growth_stage  if pred else None,
            'model_accuracy': pred.model_accuracy if pred else None,
        } if pred else None,
        'days_since_sowing': session.days_since_sowing,
        'history': list(reversed(history)),
    })


@require_http_methods(["GET"])
def api_predict_growth(request, session_id):
    """On-demand growth prediction (can pass custom sensor values)."""
    session = get_object_or_404(CropSession, id=session_id)
    try:
        # Prefer values from query string; fall back to latest sensor reading
        latest = session.sensor_readings.order_by('-timestamp').first()
        result = predict_growth(
            float(request.GET.get('ph',       latest.ph           if latest else 6.5)),
            float(request.GET.get('n',        latest.npk_nitrogen  if latest else 60)),
            float(request.GET.get('p',        latest.npk_phosphorus if latest else 30)),
            float(request.GET.get('k',        latest.npk_potassium  if latest else 50)),
            float(request.GET.get('temp',     latest.temperature   if latest else 27)),
            float(request.GET.get('humidity', latest.humidity      if latest else 65)),
            float(request.GET.get('moisture', latest.soil_moisture if latest else 45)),
            day_number  = session.days_since_sowing,
            field_acres = session.field_acres,
        )
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@require_http_methods(["GET"])
def api_growth_chart(request, session_id):
    """Chart data: actual vs predicted vs ideal curve."""
    session = get_object_or_404(CropSession, id=session_id)
    records = session.growth_records.order_by('day_number', 'slot')

    seen, days, actual, predicted, yields, health_trend, greenness_trend = {}, [], [], [], [], [], []
    for r in records:
        if r.day_number not in seen:
            seen[r.day_number] = True
            days.append(r.day_number)
            actual.append(r.actual_height_cm)
            predicted.append(r.predicted_height_cm)
            yields.append(r.predicted_yield_per_acre)
            health_trend.append(r.health_score)
            greenness_trend.append(r.actual_greenness_score)

    total_days = 60
    if session.expected_harvest:
        total_days = max(30, (session.expected_harvest - session.sowing_date).days)

    ideal_days, ideal_heights, ideal_yields = _ideal_curve(total_days)

    return JsonResponse({
        'days':             days,
        'actual_heights':   actual,
        'predicted_heights': predicted,
        'predicted_yields': yields,
        'health_trend':     health_trend,
        'greenness_trend':  greenness_trend,
        'ideal_days':       ideal_days,
        'ideal_heights':    ideal_heights,
        'ideal_yields':     ideal_yields,
        'crop':             session.crop_name,
        'field_acres':      session.field_acres,
        'total_days':       total_days,
    })


@csrf_exempt
@require_http_methods(["POST"])
def api_upload_growth_image(request, session_id):
    """Upload a plant image for a session's daily growth record."""
    session = get_object_or_404(CropSession, id=session_id)
    try:
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image provided'}, status=400)

        slot = request.POST.get('slot', 'morning' if timezone.now().hour < 14 else 'evening')

        # Save image; run disease detection alongside greenness
        plant_img = PlantImage.objects.create(
            image       = image_file,
            farmer_name = session.farmer_name,
            location    = session.location,
            notes       = request.POST.get('notes', ''),
        )
        diseased, disease_name, confidence, model_acc, treatment = predict_disease(plant_img.image.path)
        greenness = extract_greenness(plant_img.image.path)

        plant_img.disease_detected = diseased
        plant_img.disease_name     = disease_name
        plant_img.confidence       = confidence
        plant_img.model_accuracy   = model_acc
        plant_img.processed        = True
        plant_img.greenness_score  = greenness
        plant_img.save()

        # Upsert today's growth record
        today  = date.today()
        record, _ = DailyGrowthRecord.objects.get_or_create(
            session=session, date=today, slot=slot,
            defaults={'day_number': session.days_since_sowing},
        )
        record.image                 = plant_img
        record.image_uploaded        = True
        record.actual_greenness_score = greenness
        if record.day_number:
            record.actual_height_cm = round(
                10 + 70 * (1 / (1 + 2.718 ** (-0.05 * (record.day_number - 60))))
                * (greenness / 100) * 1.3, 1
            )
        # Attach disease info to growth context
        if diseased:
            record.recommendation = (record.recommendation + ' | ' if record.recommendation else '') + \
                f'Disease detected: {disease_name}'
        record.save()

        # Auto-create an alert note if disease detected
        if diseased:
            CropNote.objects.create(
                session  = session,
                category = 'alert',
                title    = f'Disease detected: {disease_name}',
                body     = f'Confidence: {confidence}%. {treatment.get("additional", "")}',
                image    = plant_img,
            )

        return JsonResponse({
            'status':           'ok',
            'image_id':         plant_img.id,
            'image_url':        plant_img.image.url,
            'disease_detected': diseased,
            'disease_name':     disease_name,
            'confidence':       confidence,
            'model_accuracy':   model_acc,
            'greenness_score':  greenness,
            'actual_height_cm': record.actual_height_cm,
            'day_number':       record.day_number,
            'slot':             slot,
            'treatment':        treatment,
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
# DISEASE DETECTION APIs  (standalone)
# ═══════════════════════════════════════════════════════════════════════════

@csrf_exempt
@require_http_methods(["POST"])
def api_disease_analyse(request):
    """Analyse a plant image — no session required."""
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

        diseased, disease_name, confidence, model_acc, treatment = predict_disease(plant_img.image.path)
        greenness = extract_greenness(plant_img.image.path)

        plant_img.disease_detected = diseased
        plant_img.disease_name     = disease_name
        plant_img.confidence       = confidence
        plant_img.model_accuracy   = model_acc
        plant_img.processed        = True
        plant_img.greenness_score  = greenness
        plant_img.save()

        return JsonResponse({
            'status':           'ok',
            'image_id':         plant_img.id,
            'image_url':        plant_img.image.url,
            'disease_detected': diseased,
            'disease_name':     disease_name,
            'confidence':       confidence,
            'model_accuracy':   model_acc,
            'greenness_score':  greenness,
            'treatment':        treatment,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def api_disease_history(request):
    """Paginated disease scan history."""
    page     = int(request.GET.get('page', 1))
    per_page = 20
    qs       = PlantImage.objects.filter(processed=True).order_by('-uploaded_at')
    total    = qs.count()
    images   = qs[(page - 1) * per_page: page * per_page]
    return JsonResponse({
        'total': total,
        'page':  page,
        'images': [{
            'id':               img.id,
            'url':              img.image.url,
            'disease_detected': img.disease_detected,
            'disease_name':     img.disease_name,
            'confidence':       img.confidence,
            'model_accuracy':   img.model_accuracy,
            'greenness_score':  img.greenness_score,
            'farmer_name':      img.farmer_name,
            'location':         img.location,
            'uploaded_at':      img.uploaded_at.strftime('%d %b %Y, %H:%M'),
        } for img in images],
    })


# ═══════════════════════════════════════════════════════════════════════════
# LEGACY  (keeps existing ESP32 firmware working with old URL)
# ═══════════════════════════════════════════════════════════════════════════

@csrf_exempt
@require_http_methods(["POST"])
def api_sensor_data_legacy(request):
    """
    Old endpoint: POST /api/sensor-data
    Routes to the active session (or first active one).
    """
    active = CropSession.objects.filter(status='active').order_by('-created_at').first()
    if not active:
        active = CropSession.objects.create(
            crop_name='Green Gram', field_acres=1.0, sowing_date=date.today()
        )
    # Reuse the session-scoped handler by temporarily patching the URL kwarg
    request.resolver_match = type('M', (), {'kwargs': {'session_id': active.id}})()
    return api_sensor_data(request, session_id=active.id)
