from django.urls import path
from . import views

urlpatterns = [
    # ── Pages ──────────────────────────────────────────────────────────────
    path('',                          views.home,              name='home'),
    path('session/<int:session_id>/', views.session_detail,    name='session_detail'),
    path('disease/',                  views.disease_detection, name='disease_detection'),

    # ── Session APIs ────────────────────────────────────────────────────────
    path('api/sessions',                          views.api_sessions,        name='api_sessions'),
    path('api/sessions/create',                   views.api_create_session,  name='api_create_session'),
    path('api/sessions/<int:session_id>/close',   views.api_close_session,   name='api_close_session'),
    path('api/sessions/<int:session_id>/delete',  views.api_delete_session,  name='api_delete_session'),

    # ── Growth monitoring APIs (scoped to a session) ────────────────────────
    path('api/sessions/<int:session_id>/sensor-data',    views.api_sensor_data,    name='api_sensor_data'),
    path('api/sessions/<int:session_id>/live-data',      views.api_live_data,      name='api_live_data'),
    path('api/sessions/<int:session_id>/growth-chart',   views.api_growth_chart,   name='api_growth_chart'),
    path('api/sessions/<int:session_id>/predict-growth', views.api_predict_growth, name='api_predict_growth'),
    path('api/sessions/<int:session_id>/upload-image',   views.api_upload_growth_image, name='api_upload_growth_image'),
    path('api/sessions/<int:session_id>/notes',          views.api_notes,          name='api_notes'),
    path('api/sessions/<int:session_id>/notes/add',      views.api_add_note,       name='api_add_note'),

    # ── Disease detection APIs (standalone, no session scope) ──────────────
    path('api/disease/analyse',  views.api_disease_analyse,  name='api_disease_analyse'),
    path('api/disease/history',  views.api_disease_history,  name='api_disease_history'),

    # ── Legacy ESP32 endpoint (keeps existing firmware working) ────────────
    path('api/sensor-data', views.api_sensor_data_legacy, name='api_sensor_data_legacy'),
]
