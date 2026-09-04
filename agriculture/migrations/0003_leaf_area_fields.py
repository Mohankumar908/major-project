from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('agriculture', '0002_cropsession_remove_growthprediction_predicted_yield_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='cropsession',
            name='camera_frame_area_cm2',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='plantimage',
            name='leaf_coverage_percent',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='plantimage',
            name='leaf_area_cm2',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='plantimage',
            name='low_confidence',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='dailygrowthrecord',
            name='actual_leaf_coverage_percent',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='dailygrowthrecord',
            name='actual_leaf_area_cm2',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
