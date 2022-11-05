# Generated by Django 4.1.2 on 2022-11-05 12:53

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Course",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("description", models.TextField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name="Gesture",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("word", models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name="Unit",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "difficulty",
                    models.CharField(
                        choices=[
                            ("1", "Beginner"),
                            ("2", "Intermediate"),
                            ("3", "Advanced"),
                        ],
                        default="1",
                        max_length=2,
                    ),
                ),
                ("gestures", models.ManyToManyField(to="sign_language_app.gesture")),
            ],
        ),
    ]
