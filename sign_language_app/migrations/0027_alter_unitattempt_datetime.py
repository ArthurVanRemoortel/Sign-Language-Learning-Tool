# Generated by Django 4.1.2 on 2023-01-12 13:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sign_language_app', '0026_unitattempt_score_alter_gestureattempt_unit_attempt'),
    ]

    operations = [
        migrations.AlterField(
            model_name='unitattempt',
            name='datetime',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]