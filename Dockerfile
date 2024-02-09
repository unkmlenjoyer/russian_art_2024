FROM python:3.10-slim

ARG PIP_DISABLE_PIP_VERSION_CHECK=1
ARG PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY ./api/main.py ./main.py
COPY ./api/config.py ./config.py
COPY ./src/initial_model_utils.py ./initial_model_utils.py
COPY ./data/weights/resnet50_tl_68.pt ./weights/resnet50_tl_68.pt

EXPOSE 8000