from django.db import models
from django.utils import timezone


class CropSession(models.Model):
    """One crop cycle from sowing to harvest."""
    STATUS_CHOICES = [
        ('active',    'Active'),
        ('completed', 'Completed'),
        ('paused',    'Paused'),
    ]
    CROP_CHOICES = [
        ('Green Gram',  'Green Gram'),
        ('Rice',        'Rice'),
        ('Wheat',       'Wheat'),
        ('Tomato',      'Tomato'),
        ('Maize',       'Maize'),
        ('Potato',      'Potato'),
        ('Soybean',     'Soybean'),
        ('Sugarcane',   'Sugarcane'),
        ('Cotton',      'Cotton'),
        ('Other',       'Other'),
    ]

    crop_name        = models.CharField(max_length=100, default='Green Gram')
    crop_variety     = models.CharField(max_length=100, blank=True)
    field_name       = models.CharField(max_length=100, blank=True)
    field_acres      = models.FloatField(default=1.0)
    farmer_name      = models.CharField(max_length=100, blank=True)
    location         = models.CharField(max_length=150, blank=True)
    sowing_date      = models.DateField()
    expected_harvest = models.DateField(null=True, blank=True)
    status           = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    notes            = models.TextField(blank=True)
    created_at       = models.DateTimeField(auto_now_add=True)
    # Calibration: real-world area (cm^2) visible in the camera's fixed frame
    # at its mounted capture distance. Set this once by measuring a known
    # object in-frame; leave null to only track leaf_coverage_percent
    # (fraction of the frame that is leaf) without an absolute cm^2 figure.
    camera_frame_area_cm2 = models.FloatField(null=True, blank=True)
    # Set ONCE at harvest with the real measured yield for this session.
    # This is the ground-truth label the yield model retrains against —
    # without it, the yield model has no real-world signal to correct on,
    # only synthetic training data.
    actual_yield_per_acre_kg = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.crop_name} — {self.field_name or 'Field'} ({self.sowing_date})"

    @property
    def days_since_sowing(self):
        return max(0, (timezone.now().date() - self.sowing_date).days)

    @property
    def is_active(self):
        return self.status == 'active'

    @property
    def progress_pct(self):
        """Percentage of expected crop cycle completed (capped at 100)."""
        if not self.expected_harvest:
            return min(100, int(self.days_since_sowing / 60 * 100))
        total = (self.expected_harvest - self.sowing_date).days
        if total <= 0:
            return 100
        return min(100, int(self.days_since_sowing / total * 100))

    @property
    def latest_health(self):
        rec = self.growth_records.exclude(health_score__isnull=True).order_by('-date', '-slot').first()
        return rec.health_score if rec else None

    @property
    def latest_stage(self):
        rec = self.growth_records.exclude(growth_stage='').order_by('-date', '-slot').first()
        return rec.growth_stage if rec else None

    @property
    def image_count_today(self):
        return self.growth_records.filter(date=timezone.now().date(), image_uploaded=True).count()


class SensorData(models.Model):
    """Raw IoT readings from ESP32."""
    session       = models.ForeignKey(
        CropSession, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='sensor_readings'
    )
    timestamp       = models.DateTimeField(auto_now_add=True)
    ph              = models.FloatField(null=True, blank=True)
    npk_nitrogen    = models.FloatField(null=True, blank=True)
    npk_phosphorus  = models.FloatField(null=True, blank=True)
    npk_potassium   = models.FloatField(null=True, blank=True)
    temperature     = models.FloatField(null=True, blank=True)
    humidity        = models.FloatField(null=True, blank=True)
    soil_moisture   = models.FloatField(null=True, blank=True)
    device_id       = models.CharField(max_length=50, default='ESP32_01')

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"Sensor @ {self.timestamp:%Y-%m-%d %H:%M} ({self.device_id})"


class PlantImage(models.Model):
    """Image uploaded for disease detection — completely standalone."""
    image            = models.ImageField(upload_to='plant_images/')
    uploaded_at      = models.DateTimeField(auto_now_add=True)
    farmer_name      = models.CharField(max_length=100, blank=True)
    location         = models.CharField(max_length=100, blank=True)
    notes            = models.TextField(blank=True)
    disease_detected = models.BooleanField(null=True)
    disease_name     = models.CharField(max_length=150, blank=True)
    confidence       = models.FloatField(null=True, blank=True)
    model_accuracy   = models.FloatField(null=True, blank=True)
    low_confidence   = models.BooleanField(default=False)
    processed        = models.BooleanField(default=False)
    greenness_score  = models.FloatField(null=True, blank=True)
    leaf_coverage_percent = models.FloatField(null=True, blank=True)
    leaf_area_cm2         = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.disease_name or 'Unprocessed'} ({self.uploaded_at:%Y-%m-%d})"


class DailyGrowthRecord(models.Model):
    """One observation slot (morning/evening) for a crop session."""
    SLOT_CHOICES = [
        ('morning', 'Morning'),
        ('evening', 'Evening'),
        ('auto',    'Auto (IoT only)'),
    ]

    session                  = models.ForeignKey(CropSession, on_delete=models.CASCADE, related_name='growth_records')
    date                     = models.DateField()
    slot                     = models.CharField(max_length=10, choices=SLOT_CHOICES, default='morning')
    day_number               = models.IntegerField(default=1)

    # Observed
    actual_height_cm         = models.FloatField(null=True, blank=True)
    actual_greenness_score   = models.FloatField(null=True, blank=True)
    actual_leaf_coverage_percent = models.FloatField(null=True, blank=True)
    actual_leaf_area_cm2         = models.FloatField(null=True, blank=True)
    image                    = models.ForeignKey(PlantImage, null=True, blank=True, on_delete=models.SET_NULL)
    image_uploaded           = models.BooleanField(default=False)

    # Predicted
    predicted_height_cm      = models.FloatField(null=True, blank=True)
    predicted_yield_per_acre = models.FloatField(null=True, blank=True)
    health_score             = models.FloatField(null=True, blank=True)
    growth_stage             = models.CharField(max_length=50, blank=True)
    recommendation           = models.TextField(blank=True)
    model_accuracy           = models.FloatField(null=True, blank=True)
    sensor_data              = models.ForeignKey(SensorData, null=True, blank=True, on_delete=models.SET_NULL)

    class Meta:
        ordering = ['date', 'slot']
        unique_together = [['session', 'date', 'slot']]

    def __str__(self):
        return f"{self.session.crop_name} Day {self.day_number} {self.slot}"


class GrowthPrediction(models.Model):
    """ML model output attached to a sensor reading."""
    sensor_data              = models.ForeignKey(SensorData, on_delete=models.CASCADE, related_name='predictions')
    timestamp                = models.DateTimeField(auto_now_add=True)
    predicted_yield_per_acre = models.FloatField(default=0)
    growth_stage             = models.CharField(max_length=50, default='Vegetative')
    health_score             = models.FloatField(default=0.0)
    recommendation           = models.TextField(blank=True)
    model_accuracy           = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']


class CropNote(models.Model):
    """Farmer observations / journal entries for a session."""
    CATEGORY_CHOICES = [
        ('observation', 'Observation'),
        ('action',      'Action Taken'),
        ('alert',       'Alert'),
        ('milestone',   'Milestone'),
    ]

    session    = models.ForeignKey(CropSession, on_delete=models.CASCADE, related_name='crop_notes')
    category   = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default='observation')
    title      = models.CharField(max_length=200)
    body       = models.TextField(blank=True)
    image      = models.ForeignKey(PlantImage, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.category}] {self.title} ({self.session.crop_name})"