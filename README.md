# Классификация клиентов Acoola по уровню лояльности

## Описание проекта

Проект направлен на разработку системы классификации клиентов компании Acoola по уровню лояльности с использованием методов машинного обучения. Система анализирует поведение клиентов, оценивает их лояльность и предоставляет как API для интеграции, так и минимальный пользовательский интерфейс для демонстрации результатов и взаимодействия с различными сегментами клиентской базы.

Основная цель — предоставить инструменты для более глубокого понимания клиентов и поддержки принятия маркетинговых решений. Проект выполнен в рамках выпускной квалификационной работы.


## Структура проекта

```
firsova_vkr/
├── .cursor/                     # Настройки Cursor IDE
├── .venv/                       # Виртуальное окружение Python
├── config/                      # Конфигурационные файлы (в перспективе)
│   ├── app/
│   └── model/
├── dataset/                     # Датасеты (например, Concept202408.csv)
├── docs/                        # Документация проекта
│   ├── business/                # Бизнес-документация
│   ├── research/                # Исследовательская документация (например, обзоры методов)
│   └── technical/               # Техническая документация
│       ├── api/                 # Документация API (в перспективе)
│       ├── architecture/        # Описание архитектуры (application_architecture.md)
│       └── visualizations_overview.md # Обзор генерируемых визуализаций
│   └── thesis_chapters/         # Материалы для ВКР
├── notebooks/                   # Jupyter notebooks (для исследовательских целей, если используются)
│   ├── exploration/
│   ├── modeling/
│   └── visualization/
├── output/                      # Выходные данные работы скриптов и моделей
│   ├── feature_importance/      # Данные и графики по важности признаков
│   │   └── visualizations/
│   ├── model_data/              # Сохраненные датасеты для обучения/теста (X_train, y_train и т.д.)
│   ├── models/                  # Сохраненные (сериализованные) модели и трансформеры
│   │   └── transformers/
│   ├── results/                 # Результаты экспериментов
│   │   └── plots/               # Графики (например, сравнение метрик моделей)
│   └── visualizations/          # Общие визуализации (например, распределение классов лояльности)
├── src/                         # Исходный код основного приложения и модулей
│   ├── api/                     # Модули API на FastAPI
│   │   ├── routes/              # Роутеры для UI и основного API
│   │   └── ...                  # (middleware, schemas - по мере развития)
│   ├── data/                    # Модули для работы с данными (загрузчики, процессоры - в перспективе)
│   ├── modeling/                # Модули, связанные с ML-моделями
│   │   ├── evaluation/          # Оценка моделей (model_evaluation.py)
│   │   ├── interpretation/      # Интерпретация моделей
│   │   ├── models/              # (Возможно, для кастомных реализаций моделей)
│   │   ├── optimization/        # Оптимизация гиперпараметров (model_optimization.py)
│   │   └── model_training.py    # Основной модуль обучения моделей
│   ├── preprocessing/           # Модули предобработки данных
│   │   ├── data_preprocessing.py # Основной модуль предобработки
│   │   └── enhanced_loyalty_features.py # Модуль создания улучшенных признаков лояльности
│   ├── static/                  # Статические файлы для UI (CSS, JS)
│   │   └── css/
│   ├── templates/               # HTML-шаблоны для UI (Jinja2)
│   │   └── partials/
│   ├── utils/                   # Вспомогательные утилиты
│   └── main.py                  # Главный файл FastAPI приложения
├── tests/                       # Тесты (в перспективе)
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── .gitattributes
├── .gitignore
├── README.md                    # Этот файл
├── requirements.txt             # Файл зависимостей Python
├── workflow_state.md            # Файл для отслеживания состояния проекта и задач
├── plan_gost.md                 # План ВКР по ГОСТ (структура глав)
└── requirements.md              # Заметки по зависимостям (исторический)
```

## Архитектура

Общее описание архитектуры приложения, включая диаграмму компонентов (C4 Model - Level 2), цели и используемые технологии, подробно изложено в документе [docs/technical/architecture/application_architecture.md](docs/technical/architecture/application_architecture.md).

Ниже представлена диаграмма компонентов из указанного документа:

```mermaid
graph TD
    accTitle: Система Классификации Лояльности Клиентов Acoola - Контейнеры
    accDescr {
        Диаграмма показывает основные контейнеры (приложения, хранилища данных) внутри Системы Классификации Лояльности Клиентов и их взаимодействия, с акцентом на API-first подход.
    }

    subgraph "Интернет"
        User[Пользователь (Аналитик/Разработчик API/Администратор)]
        ExtCRM[Внешняя CRM Система]
        ExtBI[Внешняя BI Система]
    end

    subgraph "Система Классификации Лояльности Клиентов Acoola (Развернуто в Docker)"
        AdminUIDocs["Админ UI / Документация API (FastAPI Auto-Docs + Min Admin UI)"]
        style AdminUIDocs fill:#e6e6fa,stroke:#333,stroke-width:1px
        
        APIService["API Сервис (Python FastAPI)"]
        style APIService fill:#ccf,stroke:#333,stroke-width:2px

        MLService["ML Сервис (Python + ML Libs)"]
        style MLService fill:#cfc,stroke:#333,stroke-width:2px
        
        DB["База Данных (PostgreSQL)"]
        style DB fill:#fcc,stroke:#333,stroke-width:2px
        
        DataStore["Хранилище Файлов (MinIO/Локальное)"]
        style DataStore fill:#f9c,stroke:#333,stroke-width:2px

        Queue["Очередь Задач (Celery + Redis)"]
        style Queue fill:#ffc,stroke:#333,stroke-width:2px
    end

    User --"HTTPS"--> AdminUIDocs
    AdminUIDocs --"API запросы (через Swagger/Admin UI)"--> APIService
    
    APIService --"Обработка данных, запуск ML"--> MLService
    APIService --"Чтение/Запись данных"--> DB
    APIService --"Чтение/Запись файлов (данные, модели)"--> DataStore
    APIService --"Постановка/Получение задач"--> Queue
    
    MLService --"Загрузка моделей"--> DataStore
    MLService --"Запись/Чтение данных (для ML)"--> DB
    MLService --"Выполнение задач из очереди"--> Queue

    ExtCRM --"API запросы (JSON/HTTPS)"--> APIService
    ExtBI --"API запросы (JSON/HTTPS)"--> APIService
    
    classDef user fill:#lightblue,stroke:#333,stroke-width:2px;
    classDef external fill:#lightgrey,stroke:#333,stroke-width:2px;
    class User user;
    class ExtCRM,ExtBI external;
```
<!-- Для корректного отображения диаграммы Mermaid может потребоваться плагин для вашего Markdown-просмотрщика или платформы (например, GitHub автоматически их рендерит). -->

## Установка и настройка

1.  Клонируйте репозиторий (если еще не сделали):
    ```bash
    git clone https://github.com/username/firsova_vkr.git # Замените username/firsova_vkr на актуальный URL
    cd firsova_vkr
    ```

2.  Создайте виртуальное окружение и активируйте его (рекомендуется Python 3.13 или выше):
    ```bash
    python -m venv .venv
    # Linux/Mac
    source .venv/bin/activate
    # Windows (PowerShell)
    .venv\Scripts\Activate.ps1
    # Windows (cmd.exe)
    .venv\Scripts\activate.bat
    ```

3.  Установите зависимости:
    ```bash
    pip install -r requirements.txt
    ```

4.  (Опционально) Настройте конфигурацию:
    Проект стремится минимизировать сложную настройку через конфигурационные файлы для основных сценариев. Параметры передаются через аргументы командной строки скриптов или используются значения по умолчанию.

## Использование

### Запуск основного ML пайплайна

Основной способ взаимодействия с ML-составляющей проекта — через главный скрипт `src/run_pipeline.py`.

**Пример запуска полного пайплайна:**
(Включает предобработку данных, создание признаков лояльности, обучение и оценку моделей)
```bash
python src/run_pipeline.py --input_file dataset/Concept202408.csv --output_dir output --save_intermediate
```
*   Убедитесь, что файл `Concept202408.csv` находится в директории `dataset/`. Если имя файла или путь отличаются, укажите корректный `--input_file`.

**Опции запуска пайплайна:**
Скрипт `src/run_pipeline.py` поддерживает различные флаги для управления этапами выполнения:

*   `--input_file TEXT`: Путь к исходному файлу CSV. (Обязательный)
*   `--output_dir TEXT`: Директория для сохранения результатов. По умолчанию `output`.
*   `--skip_preprocessing`: Пропустить этап предобработки данных.
*   `--skip_loyalty_features`: Пропустить этап создания признаков лояльности.
*   `--skip_model_training`: Пропустить этап обучения моделей.
*   `--tune_hyperparams`: Включить оптимизацию гиперпараметров с использованием Optuna.
*   `--n_opt_trials INTEGER`: Количество итераций для Optuna. По умолчанию 10.
*   `--cv_opt_folds INTEGER`: Количество фолдов кросс-валидации при оптимизации. По умолчанию 3.
*   `--create_ensemble`: Создать ансамблевые модели.
*   `--balance_classes TEXT`: Метод балансировки классов (например, 'smote', 'random_oversample', 'none').
*   `--save_intermediate`: Сохранять промежуточные датасеты.
*   `--random_state INTEGER`: Seed для генератора случайных чисел. По умолчанию 42.
*   `--test_size FLOAT`: Размер тестовой выборки. По умолчанию 0.2.
*   `--loyalty_metric TEXT`: Метрика для оптимизации (например, `f1_macro`, `roc_auc_ovr`). По умолчанию `f1_macro`.

**Пример:** Запуск оптимизации гиперпараметров для моделей, пропуская предобработку (если данные уже подготовлены и сохранены как `output/preprocessed_data.csv` или `output/loyalty_features_dataset.pkl`):
```bash
python src/run_pipeline.py --input_file output/loyalty_features_dataset.pkl --skip_preprocessing --skip_loyalty_features --tune_hyperparams --n_opt_trials 20
```
*Примечание: если вы пропускаете `--skip_loyalty_features`, то `--input_file` должен указывать на результат предобработки (например, `output/preprocessed_data.csv`). Если вы пропускаете и `--skip_preprocessing`, и `--skip_loyalty_features`, то `--input_file` должен указывать на датасет с уже созданными признаками лояльности (например, `output/loyalty_features_dataset.pkl`).*


### Запуск Пользовательского Интерфейса (UI) и API

Проект включает минимальный пользовательский интерфейс для демонстрации работы классификации, а также API для программного взаимодействия. Они реализованы с использованием FastAPI.

Для запуска UI и API выполните следующую команду из корневой директории проекта:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
После запуска сервер будет доступен по адресу:
*   **Пользовательский интерфейс (UI):** `http://127.0.0.1:8000/ui/`
*   **Документация API (Swagger UI):** `http://127.0.0.1:8000/docs/`
*   **Альтернативная документация API (ReDoc):** `http://127.0.0.1:8000/redoc/`

## Визуализации

Проект генерирует различные визуализации для анализа данных, оценки моделей и интерпретации результатов. Подробное описание доступных визуализаций и мест их сохранения можно найти в [docs/technical/visualizations_overview.md](docs/technical/visualizations_overview.md).

**Примеры ключевых визуализаций, которые могут быть полезны:**

*   **Распределение клиентов по сегментам лояльности:**
    <!-- Пример: output/visualizations/loyalty_categories_distribution.png -->
    <p align="center">
      <em>(Здесь можно вставить изображение графика распределения клиентов, например, `loyalty_categories_distribution.png`)</em>
    </p>

*   **Важность признаков для лучшей модели:**
    <!-- Пример: output/results/plots/feature_importance_lightgbm_ДАТА_ВРЕМЯ.png -->
    <p align="center">
      <em>(Здесь можно вставить изображение графика важности признаков для лучшей модели, например, `feature_importance_lightgbm_...png`)</em>
    </p>

*   **Сводный график SHAP values:**
    <!-- Пример: output/feature_importance/visualizations/shap_summary.png -->
    <p align="center">
      <em>(Здесь можно вставить изображение сводного графика SHAP, например, `shap_summary.png`)</em>
    </p>

## Технологический стек (основные библиотеки)

- Python 3.13+
- **Веб-фреймворк и API:**
    - FastAPI
    - Uvicorn (ASGI-сервер)
    - python-multipart (для обработки форм FastAPI)
- **Обработка данных и ML:**
    - pandas
    - numpy
    - scikit-learn
    - XGBoost
    - LightGBM
    - imbalanced-learn (для балансировки классов)
    - joblib (для сериализации моделей)
- **Оптимизация и интерпретация моделей:**
    - Optuna
    - SHAP
- **Визуализация:**
    - matplotlib
    - seaborn
    - plotly
- **Конфигурация:**
    - PyYAML (для чтения конфигурационных файлов, если используются)
- **Разработка и Тестирование (в перспективе):**
    - pytest

Полный список зависимостей указан в файле `requirements.txt`.

## Разработка

### Запуск тестов (в перспективе)
```bash
pytest tests/
```

### Форматирование кода (рекомендация)
Рекомендуется использовать `black` для форматирования кода.
```bash
black src/ tests/
```

## Лицензия

MIT (Предполагается, уточните если другая)

## Авторы

*(Здесь вы можете указать информацию об авторах проекта)*
- ...

## Благодарности

*(Если есть, кому выразить благодарность)*
