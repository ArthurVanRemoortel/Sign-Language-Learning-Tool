# Generated by Django 4.1.2 on 2023-01-02 12:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("sign_language_app", "0023_unitattempt_score_usersettings"),
    ]

    operations = [
        migrations.AlterField(
            model_name="unitattempt",
            name="score",
            field=models.IntegerField(default=0),
        ),
    ]
