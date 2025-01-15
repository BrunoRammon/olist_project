LOCAL_TAG:=$(shell date +"%Y-%m")
APP_NAME:=$(shell grep -E '^name = "[^"]+"' pyproject.toml | cut -d '"' -f 2)
LOCAL_IMAGE_NAME:=$(APP_NAME):${LOCAL_TAG}
LOCAL_CONTAINER_NAME:=$(APP_NAME)_run
SHELL := /bin/bash
ENVIROMENT_NAME:=.venv
ACTIVATE_ENVIROMENT:=source $(ENVIROMENT_NAME)/bin/activate

install_pyenv_python_version:
	pyenv install 3.11.11
set_pyenv:
	pyenv local 3.11.11
env_creation: set_pyenv
	python -m venv $(ENVIROMENT_NAME)
install_dev:
	($(ACTIVATE_ENVIROMENT) ; pip install -r requirements/dev)
install_prd:
	($(ACTIVATE_ENVIROMENT) ; pip install -r requirements/prd)
run:
	($(ACTIVATE_ENVIROMENT) ; kedro run)
start_mlflow_server:
	($(ACTIVATE_ENVIROMENT) ; mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri ./mlflow/)
kedro_docker_build:
	($(ACTIVATE_ENVIROMENT) ; kedro docker build)
kedro_docker_run:
	($(ACTIVATE_ENVIROMENT) ; kedro docker run \
							  --docker-args "--env-file ./conf/docker/.env" \
							  --env=docker)
docker_build:
	($(ACTIVATE_ENVIROMENT) ; docker build --build-arg KEDRO_UID=1000 \
										   --build-arg KEDRO_GID=1000 \
										   --build-arg BASE_IMAGE=python:3.11-slim \
										   -t olist_project .)

docker_run:
	($(ACTIVATE_ENVIROMENT) ; docker run -v ./conf/local:/home/kedro_docker/conf/local \
										 -v ./data:/home/kedro_docker/data \
										 -v ./logs:/home/kedro_docker/logs \
										 -v ./notebooks:/home/kedro_docker/notebooks \
										 -v ./references:/home/kedro_docker/references \
										 -v ./results:/home/kedro_docker/results \
										 --rm --name olist_project-run \
										 --env-file ./conf/docker/.env \
										 olist_project kedro run --env=docker)
	