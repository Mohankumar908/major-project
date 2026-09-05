from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('agriculture', '0004_merge_20260904_1547'),
    ]

    operations = [
        migrations.AddField(
            model_name='cropsession',
            name='actual_yield_per_acre_kg',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
