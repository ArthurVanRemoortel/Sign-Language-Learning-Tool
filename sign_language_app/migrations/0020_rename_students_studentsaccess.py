# Generated by Django 4.1.2 on 2022-12-31 13:17

from django.conf import settings
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('sign_language_app', '0019_teachercode_students'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Students',
            new_name='StudentsAccess',
        ),
    ]