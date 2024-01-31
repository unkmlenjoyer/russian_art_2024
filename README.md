# Russian art ML challenge 2024

# RU

### 1. Введение
------------

Задачей хакатона являлось создание системы классификации типов произведений
народного искусства Российской Федерации по фотографии.

Целевая метрика - F1 (macro). Лидерборд - 0.68 (бейзлайн - 0.58)

### 2. Структура проекта
------------

    ├── README.md                       <- Верхнеуровневое описание проекта
    ├── requirements.txt                <- Необходимые для установки библиотеки
    ├── src
    │   └── utils.py                    <- Утилиты создания и обучения модели
    │
    ├── data
    │   ├── weights                     <- Веса обученной модели соревнования
    │   ├── private_info                <- Информация о фотографиях (.csv формат)
    │   ├── train                       <- Данные для обучения
    │   └── test                        <- Данные для локальной валидации
    │
    ├── training.py                     <- Скрипт обучения модели
    └── make_submission.py              <- Скрипт генерации посылки в соревновании


### 3. Установка
-----

Для работы использовался Python 3.10.0, необходимые библиотеки есть в `requirements.txt`


# EN

### 1. Intro
-----

The task of the hackathon was to create a classification system for
people's art of the Russian Federation based on images.

The target metric is F1 (macro). Leaderboard is 0.68 (baseline 0.58)

### 2. Project structure
-----

    ├── README.md                       <- Project descriptiong (high-level)
    ├── requirements.txt                <- Libraries to install
    ├── src
    │   └── utils.py                    <- Useful functions for training
    │
    ├── data
    │   ├── weights                     <- Weights for model
    │   ├── private_info                <- Image's info
    │   ├── train                       <- Train images
    │   └── test                        <- Test images for validation
    │
    ├── training.py                     <- Script for training model
    └── make_submission.py              <- Script for generating submission

### 3. Install
-----

We used Python 3.10.0, all dependencies are in `requirements.txt` file.