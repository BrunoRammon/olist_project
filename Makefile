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