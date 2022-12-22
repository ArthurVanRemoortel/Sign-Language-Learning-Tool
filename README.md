# Sign-Language-Learning-Tool
Bachelor degree final work: An interactive learning tool for Flemish sign language using computer vision and deep learning.


## Run it locally
The easiest way to run this project is using docker + docker-compose.
### Using docker (recommended)
1. Install [Docker](https://docs.docker.com/get-docker/)
2. Install [Docker-Compose](https://docs.docker.com/compose/install/)
3. run `docker-compose up -d`
4. [Open a browser on http://localhost:8080/](http://localhost:8080/)

### Manually
This project has been developed and tested using python 3.10, but any version >= 3.5 might work as well.

1. Install `Python >= 3.10`
2. Optional: Create a virtual environment for this project.
3. Install dependencies:
   - Using pip: `> pip install requirements.txt`
   - Using [Poetry](https://python-poetry.org/): `> poetry install`
4. Set up a MySQL or MariaDB database named `learning-site`. Other databases might work depending on the django ORM compatibility.
5. Set up environment variables.
   1. `> cp env.example .env`
   2. Open the `.env` file and fill in your credentials. Tip: You can use [djecrety.ir](https://djecrety.ir/) to generate a `DJANGO_SECRET_KEY`.
6. run `> python manage.py makemigrations`
7. run `> python manage.py migrate`
8. run `> python manage.py runserver`
9. [Open a browser on http://localhost:8000/](http://localhost:8000/)

# Running the AI notebooks
I have included the notebooks that I have been using to develop the AI component of the system to use them.
Follow these steps. 
1. Complete step 1, 2, 3 of the [manual installation section.](#Manually)
2. `> pip install jupyter`
3. Some scripts will also require `ffmpeg` to be installed on your system.

## Acquiring data
Using the notebooks will require some video data to train on. You can provide your own or contact me for the data I have been using. 

## Data preparation
The `sl_ai/process_videos.py` script can be used to convert raw video footage into something more usable. This script resize the videos and some additional processing.

## Commands
- `python manage.py sync_roles`
- `python manage.py loaddad`
- `python manage.py loaddata sign_language_app/fixtures/seed.yaml`