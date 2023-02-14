#!/bin/bash

# entrypoint.sh file of Dockerfile

# Section 1- Bash options
set -o errexit
set -o pipefail
set -o nounset


echo "Waiting for postgres..."

#while ! nc -z db 5432; do
#  sleep 0.1
#done
#echo "PostgreSQL started"

cd /code

python manage.py collectstatic --noinput
python manage.py migrate
python manage.py sync_roles
gunicorn learning_site.wsgi:application -c ./config/gunicorn/prod.py

exec "$@"