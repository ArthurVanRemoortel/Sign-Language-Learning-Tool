# Sign-Language-Learning-Tool
Bachelor degree final work: An interactive learning tool for Flemish sign language using computer vision and deep learning.
The easier way to run this project is using docker + docker-compose. 

## Environment Variables

Configuration and providing credentials is achieved through environment variables. The easiest way to do this is my modifying the provided env.example file. 

1. run `cp env.example .env.docker`
2. Open `.env.docker`
3. Fill in all the fields
4. Save the `.env.docker` file.

Docker will look for the `.env.docker` file in the root directory of the project and provide all the credentials to the docker containers.

## Running Using Docker

Docker will create PostgreSQL database container and a container for the python application for you.

1. Follow the installation instruction for [Docker](https://docs.docker.com/get-docker/) and [Compose](https://docs.docker.com/compose/install/) here.
2. run `docker compose up` in the root of this project. 
3. Add the admin user: `docker exec -d [container_id] python manage.py createadmin`
4. Optional: Run `docker exec -d [container_id] python manage.py loaddata sign_language_app/fixtures/seed.yaml` to seed the database with some data.

<aside>
💡 Sometimes the PostgreSQL container takes longer to start the first, resulting in the website container failing to connect. Simply restart the containers if this happens.
</aside>

The website is now running and should be accessible at [http://localhost:8000](http://localhost:8000)

## Running Manually

This project has only been developed and tested using python 3.10.

1. Install `Python >= 3.10` using any method of choice.
2. Recommended: Create a virtual environment for this project. 
3. Install dependencies. This project has been created using the [Poetry package manager](https://python-poetry.org/). Using this package manager is optional but recommended. 
    1. Using Poetry (recommended): `poetry install`
    2. Using pip: `pip install requirements.txt` This is untested and might install different versions of packages.
4. Setup a PostgreSQL database [manually](https://www.postgresql.org/download/) or using [Docker container](https://hub.docker.com/_/postgres).
5. Create a .env file from the template. `cp env.example .env.docker`.
6. Configure your PostgreSQL credentials.
7. Run `python manage.py migrate`
8. Run `python manage.py runserver`
9. Run `python manage.py sync_roles`
10. Optional: Run `python manage.py loaddata sign_language_app/fixtures/seed.yaml` to seed the database with some data.

The website is now running and should be accessible at [http://localhost:8000](http://localhost:8000)

# Running the AI notebooks
I have included the notebooks that I have been using to develop the AI component of the system to use them.
Follow these steps. 
1. Complete steps 1, 2, 3 of the [manual installation section](#Running Manually). Do database or container are required to use the notebooks.
2. Install jupyter notebooks `pip install notebooks`
3. Optional: Older notebooks might require `ffmpeg` to be installed on your system.

## Acquiring data
Using the notebooks will require some video data to train on. You can provide your own or contact me for the data I have been using. 
