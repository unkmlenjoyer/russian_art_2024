FROM python:3.10-slim
WORKDIR /app

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY ./api/main.py ./main.py
COPY ./api/config.py ./config.py
COPY ./src/initial_model_utils.py ./initial_model_utils.py
COPY ./data/weights/resnet50_tl_68.pt ./weights/resnet50_tl_68.pt

EXPOSE 8000