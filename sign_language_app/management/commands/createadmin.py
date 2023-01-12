from django.contrib.auth.models import User
from django.core.management import BaseCommand

from learning_site import settings


# https://stackoverflow.com/a/30949410/5165250
class Command(BaseCommand):
    def handle(self, *args, **options):
        help = 'Create a an admin account using the credentials in the .env file.'

        if User.objects.count() == 0:
            if settings.DJANGO_SUPERUSER_EMAIL and settings.DJANGO_SUPERUSER_USERNAME and settings.DJANGO_SUPERUSER_PASSWORD:
                username = settings.DJANGO_SUPERUSER_USERNAME
                email = settings.DJANGO_SUPERUSER_EMAIL
                password = settings.DJANGO_SUPERUSER_PASSWORD
                print('Creating account for %s (%s)' % (username, email))
                admin = User.objects.create_superuser(email=email, username=username, password=password)
                admin.is_active = True
                admin.is_admin = True
                admin.save()
            else:
                raise Exception(f'Tried to create a superuser but the environment variables are incorrect or incomplete. '
                                f'DJANGO_SUPERUSER_EMAIL={settings.DJANGO_SUPERUSER_EMAIL}, '
                                f'DJANGO_SUPERUSER_USERNAME={settings.DJANGO_SUPERUSER_USERNAME}, '
                                f'DJANGO_SUPERUSER_PASSWORD={settings.DJANGO_SUPERUSER_PASSWORD}')
        else:
            print('Admin accounts can only be initialized if no Accounts exist')