from fastapi import APIRouter, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import re
import os
from datetime import datetime, date
import joblib # Для загрузки модели и трансформеров
import numpy as np # Для возможных операций с массивами

# --- Константы и пути ---
MODEL_PATH = "output/models/best_model.pkl"
TRANSFORMERS_BASE_DIR = "output/models/transformers"

# Ожидаемые 18 признаков для модели (из лога task_007)
EXPECTED_MODEL_FEATURES = [
    'recency_ratio', 'purchase_amount_cv', 'purchase_stability', 'recency',
    'pca_component_3', 'monetary', 'Cумма покупки_std', 'pca_component_2',
    'avg_purchase', 'pca_component_1', 'cluster', 'bonus_earning_ratio',
    'Начислено бонусов_sum', 'bonus_activity', 'Начислено бонусов_mean',
    'frequency', 'purchase_frequency', 'purchase_density'
]

# ЗАГЛУШКА: Placeholder средние значения для каждого из 18 ФИНАЛЬНЫХ признаков.
# Используются, если какой-то признак не удалось рассчитать из пользовательского ввода.
PLACEHOLDER_FINAL_FEATURE_MEANS = {
    'recency_ratio': 0.5, 'purchase_amount_cv': 0.2, 'purchase_stability': 0.8, 
    'recency': 180, 'pca_component_3': 0.0, 'monetary': 5000, 
    'Cумма покупки_std': 1000, 'pca_component_2': 0.0, 'avg_purchase': 1500, 
    'pca_component_1': 0.0, 'cluster': 2, 'bonus_earning_ratio': 0.1, 
    'Начислено бонусов_sum': 500, 'bonus_activity': 250, 'Начислено бонусов_mean': 50,
    'frequency': 10, 'purchase_frequency': 0.5, 'purchase_density': 0.05
}

# --- Загрузка основной модели --- 
model = None
class_mapping_from_model = None # Для имен классов
expected_features_from_model = None # Для имен признаков, если есть в модели

try:
    if os.path.exists(MODEL_PATH):
        loaded_data = joblib.load(MODEL_PATH)
        if isinstance(loaded_data, dict):
            model = loaded_data.get('model')
            class_mapping_from_model = loaded_data.get('class_mapping')
            expected_features_from_model = loaded_data.get('feature_names') # Имя в pkl файле было 'feature_names'
            if model:
                print(f"Модель из {MODEL_PATH} успешно загружена.")
            if class_mapping_from_model:
                print(f"Class mapping из {MODEL_PATH} успешно загружен: {class_mapping_from_model}")
            if expected_features_from_model:
                 # Если список EXPECTED_MODEL_FEATURES не совпадает с тем, что в модели, выведем предупреждение
                 # Это полезно для отладки, если пайплайн обучения изменился
                if set(EXPECTED_MODEL_FEATURES) != set(expected_features_from_model):
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Список ожидаемых признаков (EXPECTED_MODEL_FEATURES) в ui_routes.py ({len(EXPECTED_MODEL_FEATURES)}) не совпадает со списком из загруженной модели ({len(expected_features_from_model)}).")
                    print(f"Признаки из модели: {expected_features_from_model}")
                    print(f"Будет использован список из модели.")
                EXPECTED_MODEL_FEATURES = expected_features_from_model # Используем список из модели как основной
                print(f"Список признаков из модели ({len(EXPECTED_MODEL_FEATURES)}) загружен и будет использован.")
            else:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Список признаков (feature_names) не найден в {MODEL_PATH}. Будет использован EXPECTED_MODEL_FEATURES из ui_routes.py.")

        else: # Если это старый формат, где сохранялась только модель
            model = loaded_data
            print(f"Модель {MODEL_PATH} (старый формат) успешно загружена. Class mapping и feature_names не найдены.")

        if not model:
            print(f"Ошибка: не удалось извлечь модель из {MODEL_PATH}")

    else:
        print(f"Файл основной модели {MODEL_PATH} не найден.")
except Exception as e:
    print(f"Ошибка при загрузке основной модели {MODEL_PATH}: {e}")

# --- Загрузка всех трансформеров ---
transformers = {}
def load_transformer(name, path):
    try:
        if os.path.exists(path):
            transformers[name] = joblib.load(path)
            print(f'Трансформер "{name}" ({path}) успешно загружен.')
            return True
        else:
            print(f"Файл трансформера {path} для \"{name}\" не найден.")
            transformers[name] = None # Явно указываем, что трансформер не загружен
            return False
    except Exception as e:
        print(f'Ошибка при загрузке трансформера {path} для "{name}": {e}')
        transformers[name] = None
        return False

# Список трансформеров и их имена (ключи в словаре transformers)
transformer_files = {
    "monetary_vars_scaler": os.path.join(TRANSFORMERS_BASE_DIR, "monetary_vars_scaler.pkl"),
    "frequency_vars_scaler": os.path.join(TRANSFORMERS_BASE_DIR, "frequency_vars_scaler.pkl"),
    "gender_encoder": os.path.join(TRANSFORMERS_BASE_DIR, "gender_encoder.pkl"),
    "gender_mapping": os.path.join(TRANSFORMERS_BASE_DIR, "gender_mapping.pkl"),
    "top_locations_list": os.path.join(TRANSFORMERS_BASE_DIR, "top_locations_list.pkl"),
    "kmeans_scaler": os.path.join(TRANSFORMERS_BASE_DIR, "kmeans_scaler.pkl"),
    "kmeans_model": os.path.join(TRANSFORMERS_BASE_DIR, "kmeans_model.pkl"),
    "kmeans_feature_cols": os.path.join(TRANSFORMERS_BASE_DIR, "kmeans_feature_cols.pkl"),
    "pca_scaler": os.path.join(TRANSFORMERS_BASE_DIR, "pca_scaler.pkl"),
    "pca_model": os.path.join(TRANSFORMERS_BASE_DIR, "pca_model.pkl"),
    "pca_feature_cols": os.path.join(TRANSFORMERS_BASE_DIR, "pca_feature_cols.pkl"),
}

all_transformers_loaded = True
for name, path in transformer_files.items():
    if not load_transformer(name, path):
        # Если критически важный трансформер не загружен, можно установить флаг
        # Например, kmeans_model, pca_model, или основные скейлеры
        critical_transformers = ["kmeans_model", "pca_model", "monetary_vars_scaler", "frequency_vars_scaler"]
        if name in critical_transformers:
            all_transformers_loaded = False # Отметим, что не все критические трансформеры загружены
            print(f"Критически важный трансформер {name} не загружен. Функциональность может быть ограничена.")

if all_transformers_loaded:
    print("Все необходимые трансформеры успешно загружены.")
else:
    print("Не все критические трансформеры были загружены. Проверьте пути и файлы.")


templates = Jinja2Templates(directory="src/templates") 

router = APIRouter(
    tags=["User Interface"],
    responses={404: {"description": "Not found"}},
)

# --- Вспомогательные функции для обработки данных ---

def calculate_rfm(df, analysis_date_str=None):
    """ 
    Расчет Recency, Frequency, Monetary для одной строки DataFrame.
    Ожидает колонки: 'Дата последнего чека', 'Частота, раз/мес', 'Средняя сумма покупок', 'Cумма покупки' (для Monetary если нет агрегатов)
    Может также использовать 'Количество' для Frequency, если 'Частота, раз/мес' нет.
    """
    analysis_date_obj = datetime.strptime(analysis_date_str, '%Y-%m-%d').date() if analysis_date_str else date.today()
    
    # Recency: разница в днях между датой анализа и датой последней покупки
    if 'Дата последнего чека' in df.columns and pd.notnull(df.iloc[0]['Дата последнего чека']):
        # Убедимся, что дата в правильном формате
        try:
            last_purchase_date_val = pd.to_datetime(df.iloc[0]['Дата последнего чека'], errors='coerce').date()
            if pd.isna(last_purchase_date_val):
                # Попробуем другой формат, если первый не сработал
                last_purchase_date_val = pd.to_datetime(df.iloc[0]['Дата последнего чека'], format='%d/%m/%Y', errors='coerce').date()
            
            if pd.notna(last_purchase_date_val):
                df.loc[0, 'recency'] = (analysis_date_obj - last_purchase_date_val).days
            else:
                df.loc[0, 'recency'] = 365 # Заглушка, если дата невалидна
        except Exception:
             df.loc[0, 'recency'] = 365 # Заглушка при ошибке парсинга
    else:
        df.loc[0, 'recency'] = 365 # Заглушка

    # Frequency: если есть 'Частота, раз/мес', используем ее.
    # Иначе, если есть 'Количество' (покупок), можно использовать его (требует доп. логики, если это кол-во товаров)
    # Для простоты, если нет 'Частота, раз/мес', ставим 1.
    if 'Частота, раз/мес' in df.columns and pd.notnull(df.iloc[0]['Частота, раз/мес']):
        df.loc[0, 'frequency'] = df.iloc[0]['Частота, раз/мес']
    elif 'Количество' in df.columns and pd.notnull(df.iloc[0]['Количество']): # Предполагаем, что это кол-во покупок
        df.loc[0, 'frequency'] = df.iloc[0]['Количество'] 
    else:
        df.loc[0, 'frequency'] = 1 # Заглушка

    # Monetary: если есть 'Средняя сумма покупок' и 'Частота, раз/мес', то M = ССП * Частота
    # Иначе, если есть 'Cумма покупки' (для последней транзакции), используем ее как M.
    if 'Средняя сумма покупок' in df.columns and pd.notnull(df.iloc[0]['Средняя сумма покупок']) and \
       df.iloc[0]['frequency'] > 0: # Используем рассчитанную frequency
        df.loc[0, 'monetary'] = df.iloc[0]['Средняя сумма покупок'] * df.iloc[0]['frequency']
    elif 'Cумма покупки' in df.columns and pd.notnull(df.iloc[0]['Cумма покупки']):
        df.loc[0, 'monetary'] = df.iloc[0]['Cумма покупки']
    else:
        df.loc[0, 'monetary'] = 0 # Заглушка
        
    # avg_purchase - если нет, берем из 'Средняя сумма покупок' или 'Cумма покупки'
    if 'avg_purchase' not in df.columns or pd.isnull(df.iloc[0]['avg_purchase']):
        if 'Средняя сумма покупок' in df.columns and pd.notnull(df.iloc[0]['Средняя сумма покупок']):
            df.loc[0, 'avg_purchase'] = df.iloc[0]['Средняя сумма покупок']
        elif 'Cумма покупки' in df.columns and pd.notnull(df.iloc[0]['Cумма покупки']):
             df.loc[0, 'avg_purchase'] = df.iloc[0]['Cумма покупки']
        else:
            df.loc[0, 'avg_purchase'] = 0

    return df

def create_derived_features(df):
    """ 
    Создание производных признаков на основе базовых и RFM.
    Аналогично части create_enhanced_features, но для одной строки.
    """
    _df = df.copy()
    
    if 'Дата первого чека' in _df.columns and 'Дата последнего чека' in _df.columns and \
       pd.notnull(_df.iloc[0]['Дата первого чека']) and pd.notnull(_df.iloc[0]['Дата последнего чека']):
        try:
            date_first_check = pd.to_datetime(_df.iloc[0]['Дата первого чека'], errors='coerce').date()
            date_last_check = pd.to_datetime(_df.iloc[0]['Дата последнего чека'], errors='coerce').date()
            
            if pd.isna(date_first_check):
                date_first_check = pd.to_datetime(_df.iloc[0]['Дата первого чека'], format='%d/%m/%Y', errors='coerce').date()
            if pd.isna(date_last_check):
                date_last_check = pd.to_datetime(_df.iloc[0]['Дата последнего чека'], format='%d/%m/%Y', errors='coerce').date()

            if pd.notna(date_first_check) and pd.notna(date_last_check):
                activity_period = (date_last_check - date_first_check).days + 1
                _df.loc[0, 'activity_period'] = max(activity_period, 1)
            else:
                _df.loc[0, 'activity_period'] = 1
        except Exception:
            _df.loc[0, 'activity_period'] = 1
    else:
        _df.loc[0, 'activity_period'] = 1

    if pd.notnull(_df.iloc[0].get('recency')) and pd.notnull(_df.iloc[0].get('activity_period')) and _df.iloc[0]['activity_period'] > 0:
        _df.loc[0, 'recency_ratio'] = _df.iloc[0]['recency'] / _df.iloc[0]['activity_period']
    else:
        _df.loc[0, 'recency_ratio'] = 0.5

    if pd.notnull(_df.iloc[0].get('frequency')) and pd.notnull(_df.iloc[0].get('activity_period')) and _df.iloc[0]['activity_period'] > 0:
        _df.loc[0, 'purchase_density'] = _df.iloc[0]['frequency'] / _df.iloc[0]['activity_period']
    else:
        _df.loc[0, 'purchase_density'] = 0.05

    _df.loc[0, 'purchase_frequency'] = _df.iloc[0].get('frequency', 1)

    _df.loc[0, 'Начислено бонусов_sum'] = _df.iloc[0].get('Начислено бонусов', 0) if pd.notnull(_df.iloc[0].get('Начислено бонусов')) else 0
    _df.loc[0, 'Начислено бонусов_mean'] = _df.iloc[0]['Начислено бонусов_sum']
    
    sum_spisano_bonusov = _df.iloc[0].get('Списано бонусов', 0) if pd.notnull(_df.iloc[0].get('Списано бонусов')) else 0

    if pd.notnull(_df.iloc[0].get('monetary')) and _df.iloc[0].get('monetary', 0) > 0:
        _df.loc[0, 'bonus_earning_ratio'] = _df.iloc[0]['Начислено бонусов_sum'] / _df.iloc[0]['monetary']
    else:
        _df.loc[0, 'bonus_earning_ratio'] = 0

    _df.loc[0, 'bonus_activity'] = (_df.iloc[0]['Начислено бонусов_sum'] + sum_spisano_bonusov) / 2

    _df.loc[0, 'Cумма покупки_std'] = 0
    _df.loc[0, 'purchase_amount_cv'] = 0
    _df.loc[0, 'purchase_stability'] = 1

    return _df

# --- Основные эндпоинты UI ---
@router.get("/", response_class=HTMLResponse)
async def get_main_ui_page(request: Request):
    # Логика для главной страницы
    # Например, передача информации о модели, общая статистика
    return templates.TemplateResponse("index.html", {"request": request, "page_title": "Анализ лояльности клиентов"})

@router.get("/model-info", response_class=HTMLResponse)
async def get_model_info(request: Request):
    model_data = {
        "name": "N/A",
        "version": "N/A", # Версию пока не извлекаем, можно добавить если есть источник
        "f1_macro": "N/A",
        "trained_date": "N/A"
    }
    log_file_path = "output/results/task_007_optimization_run_log.txt"
    model_file_path = "output/models/best_model.pkl"

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # Извлечение имени лучшей модели
        name_match = re.search(r"Лучшая модель по F1-macro: (\w+)", log_content)
        if name_match:
            model_data["name"] = name_match.group(1)

        # Извлечение F1-macro
        f1_match = re.search(r"Лучшая модель по F1-macro: \w+\s*(?:(?:\n.*?)+?-\s*f1_macro:\s*([\d\.]+))", log_content, re.MULTILINE)
        if f1_match:
            model_data["f1_macro"] = f1_match.group(1)
        else: # Попробуем другой паттерн, если первый не сработал (на случай небольших изменений в логе)
            f1_alt_match = re.search(r"Оптимизация для lightgbm завершена\. Лучший F1 \(macro\): ([\d\.]+)", log_content)
            if f1_alt_match and model_data["name"].lower() == "lightgbm": # Убедимся, что это для lightgbm
                 model_data["f1_macro"] = f1_alt_match.group(1)


        # Извлечение даты из строки "Выполнение завершено в YYYY-MM-DD HH:MM:SS"
        date_match = re.search(r"Выполнение завершено в (\d{4}-\d{2}-\d{2})", log_content)
        if date_match:
            model_data["trained_date"] = date_match.group(1)
        
        # Если из лога дату не удалось извлечь, или для большей точности,
        # можно использовать дату модификации файла модели
        if os.path.exists(model_file_path) and model_data["trained_date"] == "N/A":
            timestamp = os.path.getmtime(model_file_path)
            model_data["trained_date"] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

    except Exception as e:
        print(f"Ошибка при чтении информации о модели из лога: {e}")
        # Данные по умолчанию останутся N/A

    return templates.TemplateResponse("partials/model_info.html", {"request": request, "model": model_data})

@router.get("/loyalty-distribution", response_class=HTMLResponse)
async def get_loyalty_distribution_data(request: Request):
    distribution_data = {
        "labels": [],
        "values": []
    }
    log_file_path = "output/results/task_007_optimization_run_log.txt"

    # Соответствие классов реальным именам сегментов (нужно будет уточнить или сделать настраиваемым)
    # Это ПРЕДПОЛОЖЕНИЕ на основе типичного порядка от "менее лояльных" к "более лояльным"
    # или по частоте встречаемости, как было в логе ранее. Лучше иметь явное сопоставление.
    class_to_segment_name = {
        "Класс 0": "Низколояльные",       # (38.18%)
        "Класс 1": "Умеренно лояльные", # (21.82%)
        "Класс 2": "Лояльные",            # (20.01%)
        "Класс 3": "Высоколояльные"     # (20.00%)
    }

    # Используем class_mapping_from_model если он был загружен
    if class_mapping_from_model:
        # class_mapping_from_model это {0: 'Name0', 1: 'Name1', ...}
        # Нам нужно { 'Name0': count, 'Name1': count, ... } для графика или просто передать в шаблон
        # Лог парсится по "Класс X", поэтому нам нужно сначала смапить "Класс X" на реальное имя из class_mapping_from_model
        # а затем уже парсить количества.
        # Однако, если лог содержит точные имена, которые есть в class_mapping_from_model, то можно напрямую.
        # Проще всего передать class_mapping_from_model в шаблон и там решить, как отображать.
        # Здесь для простоты оставим парсинг как был, но для отображения в HTML будем использовать class_mapping_from_model.
        pass # class_mapping_from_model будет передан в шаблон

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # Ищем блок с распределением целевой переменной
        distribution_block_match = re.search(r"Распределение целевой переменной:\s*\n((?:\s*-\s*Класс \d+: \d+ \([\d\.]+%\)\s*\n?)+)", log_content, re.MULTILINE)
        
        if distribution_block_match:
            block_content = distribution_block_match.group(1)
            lines = block_content.strip().split('\n')
            for line in lines:
                line = line.strip()
                match = re.search(r"(Класс \d+): (\d+) \(([\d\.]+)%\)", line)
                if match:
                    class_label = match.group(1)
                    count = int(match.group(2))
                    # Используем реальное имя сегмента, если оно есть в нашем словаре
                    segment_name = class_to_segment_name.get(class_label, class_label) 
                    distribution_data["labels"].append(segment_name)
                    distribution_data["values"].append(count)
            
            # Если данные были найдены и добавлены, но result пустой, значит что-то пошло не так при парсинге.
            # В таком случае, можно оставить данные пустыми, чтобы шаблон отобразил сообщение об ошибке/отсутствии данных.
            if not distribution_data["labels"] or not distribution_data["values"]:
                 distribution_data = {"labels": [], "values": []} # Сброс, если парсинг не удался
                 if class_mapping_from_model:
                    # Попробуем заполнить из class_mapping_from_model, если лог не дал результатов
                    # Это предполагает, что class_mapping_from_model содержит имена, которые мы хотим видеть как labels
                    # А значения (values) мы из лога все равно не вытащили.
                    # Лучше просто передать class_mapping_from_model и пусть шаблон решает.
                    pass 

        else:
            print("Блок с распределением целевой переменной не найден в логе.")
            # distribution_data останется пустым

    except Exception as e:
        print(f"Ошибка при чтении информации о распределении лояльности: {e}")
        # distribution_data останется пустым, чтобы шаблон отобразил сообщение по умолчанию

    # Если после всех попыток данных нет, шаблон отобразит "Данные ... не загружены"
    return templates.TemplateResponse("partials/loyalty_distribution.html", 
                                    {"request": request, 
                                     "distribution": distribution_data, 
                                     "class_mapping": class_mapping_from_model})

@router.get("/classification-results", response_class=HTMLResponse)
async def get_classification_results(request: Request):
    # Логика для отображения таблицы с результатами классификации
    # TODO: Загрузить демонстрационные результаты классификации (например, из CSV)
    # TODO: Преобразовать в список словарей или использовать pandas to_html
    results_data = [
        {"client_id": "123", "segment": "Лояльные", "score": 0.85},
        {"client_id": "456", "segment": "Высоколояльные", "score": 0.95},
        # ... другие клиенты
    ] # Пример
    return templates.TemplateResponse("partials/classification_results_table.html", {"request": request, "results": results_data})

@router.post("/classify", response_class=HTMLResponse)
async def classify_data(
    request: Request,
    file: UploadFile = File(None), # Для загрузки CSV
    # Поля для ручного ввода (примерный набор, нужно определить точнее)
    manual_recency: float = Form(None), 
    manual_frequency: float = Form(None), 
    manual_monetary: float = Form(None), 
    manual_avg_purchase: float = Form(None),
    manual_bonus_sum_earned: float = Form(None),
    manual_bonus_sum_spent: float = Form(None),
    manual_activity_period_days: float = Form(None), # Это поле можно убрать, если рассчитываем из дат
    manual_purchase_std_dev: float = Form(None), # Это поле убираем, для одной транзакции std=0

    # Новые поля из CSV
    manual_client_id: str = Form(None), # Клиент ID
    manual_gender: str = Form(None), # Пол
    manual_card_issue_store: str = Form(None), # Точка продаж где оформлена карта
    manual_card_issue_date: str = Form(None), # Дата оформления карты (YYYY-MM-DD)
    manual_purchase_date: str = Form(None), # Дата покупки (YYYY-MM-DD) - используется как Дата последнего чека
    manual_purchase_store: str = Form(None), # Точка продаж
    # manual_product_name: str = Form(None), # Название товара - пока не используем напрямую
    manual_purchase_amount: float = Form(None), # Cумма покупки
    manual_quantity: int = Form(None), # Количество (предполагаем, что это частота для одной записи)
    # manual_external_discount: float = Form(None), # Скидка внешняя - пока не используем
    # manual_first_check_date: str = Form(None), # Дата первого чека (YYYY-MM-DD)
    # manual_accumulated_balance: float = Form(None), # Баланс накопленный - пока не используем
    # manual_gift_balance: float = Form(None) # Баланс подарочный - пока не используем
):
    results = []
    error_message = None
    info_message = None # Для дополнительной информации пользователю
    input_method = ""
    
    # DataFrame для ВСЕХ признаков, которые мы смогли извлечь/вычислить из пользовательского ввода
    # или CSV, включая промежуточные.
    # Этот DataFrame будет содержать результаты для всех строк (если CSV) или одной строки (если ручной ввод)
    # перед финальным формированием признаков для модели.
    all_processed_data_df = pd.DataFrame()


    if not model:
        error_message = "Модель классификации не загружена. Обратитесь к администратору."
        return templates.TemplateResponse("partials/classification_result_display.html",
                                        {"request": request, "error": error_message, "input_method": "Ошибка системы"})

    if file and file.filename:
        input_method = f"Файл: {file.filename}"
        try:
            contents = await file.read()
            raw_df_from_csv = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            
            if raw_df_from_csv.empty:
                error_message = "Файл CSV пуст или не удалось извлечь данные."
            else:
                processed_rows_list = [] # Список для сбора обработанных строк (в виде словарей или DataFrame)
                num_rows = len(raw_df_from_csv)
                print(f"Начата обработка {num_rows} строк из CSV файла...")

                for i, raw_row_series in raw_df_from_csv.iterrows():
                    print(f"Обработка строки {i+1}/{num_rows} из CSV...")
                    current_input_df = pd.DataFrame([raw_row_series]) # Оборачиваем серию в DataFrame для совместимости с функциями

                    # --- Начало блока обработки одной строки (аналогично ручному вводу) ---
                    # 1. Базовая предобработка (извлечение Client ID, Пол, Точка продаж)
                    # Клиент ID (если есть колонка)
                    client_id_col_names = ['Клиент', 'client_id', 'ID Клиента'] # Возможные имена колонки
                    found_client_id_col = None
                    for name in client_id_col_names:
                        if name in current_input_df.columns:
                            found_client_id_col = name
                            break
                    current_input_df['Клиент'] = current_input_df[found_client_id_col].iloc[0] if found_client_id_col else f"csv_user_{i+1}"
                    
                    # Очистка и приведение типов для ключевых полей из CSV
                    # (добавлено из блока ручного ввода для консистентности)
                    if 'Cумма покупки' in current_input_df.columns:
                        current_input_df['Cумма покупки'] = pd.to_numeric(current_input_df['Cумма покупки'], errors='coerce').fillna(0)
                    if 'Количество' in current_input_df.columns:
                        current_input_df['Количество'] = pd.to_numeric(current_input_df['Количество'], errors='coerce').fillna(1)
                    if 'Начислено бонусов' in current_input_df.columns:
                        current_input_df['Начислено бонусов'] = pd.to_numeric(current_input_df['Начислено бонусов'], errors='coerce').fillna(0)
                    if 'Списано бонусов' in current_input_df.columns:
                        current_input_df['Списано бонусов'] = pd.to_numeric(current_input_df['Списано бонусов'], errors='coerce').fillna(0)
                    
                    # Пол
                    gender_col_names = ['Пол', 'gender']
                    found_gender_col = None
                    for name in gender_col_names:
                        if name in current_input_df.columns:
                            found_gender_col = name
                            break

                    if transformers.get("gender_encoder") and found_gender_col:
                        try:
                            gender_map = transformers.get("gender_mapping", {})
                            default_gender_str = next(iter(gender_map.keys())) if gender_map else "Не указан"
                            current_input_df[found_gender_col] = current_input_df[found_gender_col].fillna(default_gender_str).astype(str)
                            
                            # Применяем mapping перед transform, если значения не совпадают с теми, на которых обучался энкодер
                            current_input_df[found_gender_col] = current_input_df[found_gender_col].apply(
                                lambda x: x if x in gender_map else default_gender_str
                            )
                            current_input_df['Пол_encoded'] = transformers["gender_encoder"].transform(current_input_df[[found_gender_col]])[0]
                        except Exception as e:
                            print(f"Ошибка кодирования пола для строки {i+1} CSV: {e}. Используется заглушка 0.")
                            current_input_df['Пол_encoded'] = 0
                    else:
                        current_input_df['Пол_encoded'] = 0

                    # Точка продаж (OHE)
                    store_col_names = ['Точка продаж', 'store', 'Purchase Store']
                    found_store_col = None
                    for name in store_col_names:
                        if name in current_input_df.columns:
                            found_store_col = name
                            break
                    
                    if transformers.get("top_locations_list") and found_store_col:
                        top_locations = transformers["top_locations_list"]
                        current_location = str(current_input_df.iloc[0][found_store_col]) if pd.notnull(current_input_df.iloc[0][found_store_col]) else "Unknown"
                        for loc in top_locations:
                            current_input_df[f'Точка продаж_{loc}'] = 1 if current_location == loc else 0
                        if current_location not in top_locations:
                            current_input_df['Точка продаж_Other'] = 1
                        else:
                            current_input_df['Точка продаж_Other'] = 0
                    else:
                        if transformers.get("top_locations_list"):
                            for loc in transformers.get("top_locations_list", []): current_input_df[f'Точка продаж_{loc}'] = 0
                        current_input_df['Точка продаж_Other'] = 1
                    
                    # 2. Расчет RFM и производных признаков
                    # Для RFM нужна 'Дата последнего чека'. Предположим, что в CSV есть колонка с датой покупки.
                    purchase_date_col_names = ['Дата последнего чека', 'Дата покупки', 'Purchase Date', 'Date']
                    found_purchase_date_col = None
                    for name in purchase_date_col_names:
                        if name in current_input_df.columns:
                            found_purchase_date_col = name
                            break
                    
                    analysis_date_for_rfm_csv = None
                    if found_purchase_date_col and pd.notnull(current_input_df.iloc[0][found_purchase_date_col]):
                        try:
                            # Попытка преобразовать дату и использовать ее. Если не получится, RFM может быть неточным.
                            analysis_date_for_rfm_csv = pd.to_datetime(current_input_df.iloc[0][found_purchase_date_col], errors='coerce').strftime('%Y-%m-%d')
                            current_input_df['Дата последнего чека'] = analysis_date_for_rfm_csv # Обновляем для calculate_rfm
                        except Exception as e:
                            print(f"Не удалось извлечь дату покупки из CSV для строки {i+1}: {current_input_df.iloc[0][found_purchase_date_col]}. Ошибка: {e}")
                    
                    # Если 'Дата первого чека' не предоставлена, но есть 'Дата последнего чека',
                    # используем ее для расчета 'activity_period' в create_derived_features
                    first_check_date_col_names = ['Дата первого чека', 'First Purchase Date']
                    found_first_check_date_col = None
                    for name in first_check_date_col_names:
                        if name in current_input_df.columns and pd.notnull(current_input_df.iloc[0][name]):
                            found_first_check_date_col = name
                            break
                    if not found_first_check_date_col and found_purchase_date_col and pd.notnull(current_input_df.iloc[0][found_purchase_date_col]):
                         current_input_df['Дата первого чека'] = current_input_df.iloc[0][found_purchase_date_col]
                    
                    current_input_df = calculate_rfm(current_input_df, analysis_date_str=analysis_date_for_rfm_csv) # None если дата не извлеклась
                    current_input_df = create_derived_features(current_input_df)
                    
                    # На данном этапе в current_input_df есть базовые + RFM + производные признаки для одной строки CSV

                    # 3. Подготовка данных для KMeans и применение KMeans
                    if transformers.get("kmeans_model") and transformers.get("kmeans_scaler") and transformers.get("kmeans_feature_cols"):
                        kmeans_cols = transformers["kmeans_feature_cols"]
                        kmeans_input_data_row = pd.DataFrame()
                        missing_kmeans_cols = []
                        for col in kmeans_cols:
                            if col in current_input_df.columns:
                                kmeans_input_data_row[col] = current_input_df[col]
                            else:
                                kmeans_input_data_row[col] = PLACEHOLDER_FINAL_FEATURE_MEANS.get(col, 0)
                                missing_kmeans_cols.append(col)
                        if missing_kmeans_cols:
                            print(f"CSV строка {i+1}: для KMeans отсутствовали колонки: {missing_kmeans_cols}. Заполнены заглушками.")
                        
                        kmeans_input_data_row.fillna(kmeans_input_data_row.mean(numeric_only=True), inplace=True)
                        try:
                            kmeans_scaled_row = transformers["kmeans_scaler"].transform(kmeans_input_data_row[kmeans_cols])
                            cluster_label_row = transformers["kmeans_model"].predict(kmeans_scaled_row)
                            current_input_df['cluster'] = cluster_label_row[0]
                        except Exception as e:
                            print(f"CSV строка {i+1}: Ошибка KMeans: {e}. Кластер = заглушка.")
                            current_input_df['cluster'] = PLACEHOLDER_FINAL_FEATURE_MEANS.get('cluster', 2)
                    else:
                        current_input_df['cluster'] = PLACEHOLDER_FINAL_FEATURE_MEANS.get('cluster', 2)

                    # 4. Подготовка данных для PCA и применение PCA
                    if transformers.get("pca_model") and transformers.get("pca_scaler") and transformers.get("pca_feature_cols"):
                        pca_cols = transformers["pca_feature_cols"]
                        n_pca_components = transformers["pca_model"].n_components_ if hasattr(transformers["pca_model"], 'n_components_') else 3
                        pca_input_data_row = pd.DataFrame()
                        missing_pca_cols = []
                        for col in pca_cols:
                            if col in current_input_df.columns:
                                pca_input_data_row[col] = current_input_df[col]
                            else:
                                pca_input_data_row[col] = PLACEHOLDER_FINAL_FEATURE_MEANS.get(col, 0)
                                missing_pca_cols.append(col)
                        if missing_pca_cols:
                            print(f"CSV строка {i+1}: для PCA отсутствовали колонки: {missing_pca_cols}. Заполнены заглушками.")

                        pca_input_data_row.fillna(pca_input_data_row.mean(numeric_only=True), inplace=True)
                        try:
                            pca_scaled_row = transformers["pca_scaler"].transform(pca_input_data_row[pca_cols])
                            pca_components_values_row = transformers["pca_model"].transform(pca_scaled_row)
                            for comp_idx in range(n_pca_components):
                                current_input_df[f'pca_component_{comp_idx+1}'] = pca_components_values_row[0, comp_idx]
                        except Exception as e:
                            print(f"CSV строка {i+1}: Ошибка PCA: {e}. Компоненты = заглушки.")
                            for comp_idx in range(n_pca_components):
                                current_input_df[f'pca_component_{comp_idx+1}'] = PLACEHOLDER_FINAL_FEATURE_MEANS.get(f'pca_component_{comp_idx+1}', 0)
                    else:
                        n_pca_components = 3 
                        for comp_idx in range(n_pca_components):
                            current_input_df[f'pca_component_{comp_idx+1}'] = PLACEHOLDER_FINAL_FEATURE_MEANS.get(f'pca_component_{comp_idx+1}', 0)
                    # --- Конец блока обработки одной строки ---
                    processed_rows_list.append(current_input_df)
                
                if processed_rows_list:
                    all_processed_data_df = pd.concat(processed_rows_list, ignore_index=True)
                    info_message = f"Данные из файла CSV ({num_rows} строк) обработаны. Применены расчет RFM, производных признаков, KMeans и PCA для каждой строки."
                    if not all_transformers_loaded:
                        info_message += " Некоторые трансформеры могли быть не загружены, что повлияло на расчеты (использовались заглушки)."
                else:
                    error_message = "Не удалось обработать строки из CSV файла."

        except Exception as e:
            error_message = f"Ошибка обработки файла CSV: {str(e)}"
            import traceback
            traceback.print_exc()
            results = []
            all_processed_data_df = pd.DataFrame() # Сбрасываем, если была ошибка

    elif any(v is not None for v in [
        manual_client_id, manual_gender, manual_card_issue_store, manual_card_issue_date,
        manual_purchase_date, manual_purchase_store, manual_purchase_amount, manual_quantity,
        manual_recency, manual_frequency, manual_monetary, manual_avg_purchase, # Старые поля RFM, если используются
        manual_bonus_sum_earned, manual_bonus_sum_spent, manual_activity_period_days 
        # manual_purchase_std_dev - убрано, т.к. для одной транзакции = 0
    ]):
        input_method = "Ручной ввод"
        try:
            # 1. Собираем все сырые данные из формы в DataFrame
            raw_manual_data = {
                'Клиент': [manual_client_id if manual_client_id else "manual_user_01"], # Добавим ID по умолчанию
                'Пол': [manual_gender],
                'Точка продаж, где оформлена карта': [manual_card_issue_store],
                'Дата оформления карты': [manual_card_issue_date],
                'Дата последнего чека': [manual_purchase_date], # Используем как дату последней покупки
                'Точка продаж': [manual_purchase_store],
                'Cумма покупки': [manual_purchase_amount],
                'Количество': [manual_quantity], # Может использоваться для Frequency
                'Начислено бонусов': [manual_bonus_sum_earned],
                'Списано бонусов': [manual_bonus_sum_spent],
                # Поля, которые могут быть переданы напрямую или рассчитаны (старый интерфейс)
                'recency': [manual_recency],
                'frequency': [manual_frequency],
                'monetary': [manual_monetary],
                'avg_purchase': [manual_avg_purchase],
                'activity_period': [manual_activity_period_days] # Если передано напрямую
            }
            # Добавим 'Дата первого чека' если она не передана, но есть 'Дата последнего чека'
            # Для простоты, если нет 'activity_period_days', будем считать, что первая покупка = последней.
            if manual_purchase_date and not manual_activity_period_days:
                 raw_manual_data['Дата первого чека'] = [manual_purchase_date]


            current_input_df = pd.DataFrame.from_dict(raw_manual_data)

            # 2. Базовая предобработка (аналогично основному пайплайну, но упрощенно)
            # Обработка пропусков в ключевых полях перед расчетами
            if 'Cумма покупки' in current_input_df.columns:
                current_input_df['Cумма покупки'] = pd.to_numeric(current_input_df['Cумма покупки'], errors='coerce').fillna(0)
            if 'Количество' in current_input_df.columns:
                current_input_df['Количество'] = pd.to_numeric(current_input_df['Количество'], errors='coerce').fillna(1) # хотя бы 1
            if 'Начислено бонусов' in current_input_df.columns:
                current_input_df['Начислено бонусов'] = pd.to_numeric(current_input_df['Начислено бонусов'], errors='coerce').fillna(0)
            if 'Списано бонусов' in current_input_df.columns:
                current_input_df['Списано бонусов'] = pd.to_numeric(current_input_df['Списано бонусов'], errors='coerce').fillna(0)

            # Пол
            if transformers.get("gender_encoder") and 'Пол' in current_input_df.columns:
                try:
                    # Замена пропусков или неизвестных значений перед кодированием
                    # Используем mapping, если он есть, чтобы заменить на известное значение или спец.категорию
                    gender_map = transformers.get("gender_mapping", {}) # { 'М': 0, 'Ж': 1 }
                    # Инвертируем для поиска ключа по значению, или просто используем первый ключ как дефолт
                    default_gender_str = next(iter(gender_map.keys())) if gender_map else "Не указан"
                    
                    current_input_df['Пол'] = current_input_df['Пол'].apply(
                        lambda x: x if x in gender_map else default_gender_str
                    )
                    current_input_df['Пол_encoded'] = transformers["gender_encoder"].transform(current_input_df[['Пол']])
                except Exception as e:
                    print(f"Ошибка кодирования пола: {e}. Используется заглушка 0.")
                    current_input_df['Пол_encoded'] = 0 # Заглушка
            else:
                current_input_df['Пол_encoded'] = 0 # Заглушка, если нет энкодера

            # Точка продаж (OHE на основе top_locations_list)
            if transformers.get("top_locations_list") and 'Точка продаж' in current_input_df.columns:
                top_locations = transformers["top_locations_list"]
                current_location = current_input_df.iloc[0]['Точка продаж']
                for loc in top_locations:
                    current_input_df[f'Точка продаж_{loc}'] = 1 if current_location == loc else 0
                if current_location not in top_locations:
                     current_input_df['Точка продаж_Other'] = 1
                else:
                    current_input_df['Точка продаж_Other'] = 0
            else: # Заглушки, если нет списка
                if transformers.get("top_locations_list"):
                    for loc in transformers.get("top_locations_list", []): current_input_df[f'Точка продаж_{loc}'] = 0
                current_input_df['Точка продаж_Other'] = 1


            # 3. Расчет RFM и производных признаков
            analysis_date_for_rfm = manual_purchase_date # Используем дату покупки как дату анализа для RFM
            current_input_df = calculate_rfm(current_input_df, analysis_date_str=analysis_date_for_rfm)
            current_input_df = create_derived_features(current_input_df)
            
            # Сохраняем то, что получилось на этом этапе
            # Для ручного ввода all_processed_data_df будет содержать одну строку
            all_processed_data_df = current_input_df.copy() 

            # 4. Подготовка данных для KMeans и применение KMeans
            if transformers.get("kmeans_model") and transformers.get("kmeans_scaler") and transformers.get("kmeans_feature_cols"):
                kmeans_cols = transformers["kmeans_feature_cols"]
                kmeans_input_data = pd.DataFrame()
                missing_kmeans_cols = []
                for col in kmeans_cols:
                    if col in current_input_df.columns:
                        kmeans_input_data[col] = current_input_df[col]
                    else:
                        kmeans_input_data[col] = PLACEHOLDER_FINAL_FEATURE_MEANS.get(col, 0)
                        missing_kmeans_cols.append(col)
                if missing_kmeans_cols:
                    print(f"Для KMeans отсутствовали колонки: {missing_kmeans_cols}. Заполнены заглушками.")

                kmeans_input_data.fillna(kmeans_input_data.mean(numeric_only=True), inplace=True) 
                
                try:
                    kmeans_scaled = transformers["kmeans_scaler"].transform(kmeans_input_data[kmeans_cols]) 
                    cluster_labels = transformers["kmeans_model"].predict(kmeans_scaled)
                    all_processed_data_df['cluster'] = cluster_labels[0] # Берем первый элемент, т.к. одна строка (для ручного)
                except Exception as e:
                    print(f"Ошибка при применении KMeans: {e}. Используется заглушка для кластера.")
                    all_processed_data_df['cluster'] = PLACEHOLDER_FINAL_FEATURE_MEANS.get('cluster', 2) 
            else:
                print("KMeans модель, скейлер или список признаков не загружены. Используется заглушка для кластера.")
                # all_processed_data_df уже содержит processed_input_df, где кластер был установлен или будет установлен ниже
                if 'cluster' not in all_processed_data_df.columns:
                     all_processed_data_df['cluster'] = PLACEHOLDER_FINAL_FEATURE_MEANS.get('cluster', 2)

            # 5. Подготовка данных для PCA и применение PCA
            if transformers.get("pca_model") and transformers.get("pca_scaler") and transformers.get("pca_feature_cols"):
                pca_cols = transformers["pca_feature_cols"]
                n_pca_components = transformers["pca_model"].n_components_ if hasattr(transformers["pca_model"], 'n_components_') else 3

                pca_input_data = pd.DataFrame()
                missing_pca_cols = []
                for col in pca_cols:
                    if col in current_input_df.columns:
                        pca_input_data[col] = current_input_df[col]
                    else:
                        pca_input_data[col] = PLACEHOLDER_FINAL_FEATURE_MEANS.get(col, 0)
                        missing_pca_cols.append(col)
                if missing_pca_cols:
                    print(f"Для PCA отсутствовали колонки: {missing_pca_cols}. Заполнены заглушками.")
                
                pca_input_data.fillna(pca_input_data.mean(numeric_only=True), inplace=True)

                try:
                    pca_scaled = transformers["pca_scaler"].transform(pca_input_data[pca_cols])
                    pca_components_values = transformers["pca_model"].transform(pca_scaled)
                    
                    for i in range(n_pca_components):
                        all_processed_data_df[f'pca_component_{i+1}'] = pca_components_values[0, i] # Для ручного ввода
                except Exception as e:
                    print(f"Ошибка при применении PCA: {e}. Используются заглушки для PCA компонент.")
                    for i in range(n_pca_components):
                        all_processed_data_df[f'pca_component_{i+1}'] = PLACEHOLDER_FINAL_FEATURE_MEANS.get(f'pca_component_{i+1}', 0)
            else:
                print("PCA модель, скейлер или список признаков не загружены. Используются заглушки для PCA компонент.")
                n_pca_components = 3 # Предполагаем 3 по умолчанию для заглушек
                for i in range(n_pca_components):
                    # all_processed_data_df уже содержит processed_input_df, где pca компоненты были установлены или будут
                    if f'pca_component_{i+1}' not in all_processed_data_df.columns:
                        all_processed_data_df[f'pca_component_{i+1}'] = PLACEHOLDER_FINAL_FEATURE_MEANS.get(f'pca_component_{i+1}', 0)

            # info_message уже был установлен в блоке try ручного ввода
            # Обновляем его, если нужно, на основе all_processed_data_df
            if not error_message:
                 info_message = "Данные ручного ввода обработаны. Применены расчет RFM, производных признаков, KMeans и PCA."
                 if not all_transformers_loaded: 
                    info_message += " Некоторые трансформеры могли быть не загружены, что повлияло на расчеты (использовались заглушки)."

        except Exception as e:
            error_message = f"Ошибка обработки данных ручного ввода: {str(e)}"
            import traceback
            traceback.print_exc()
            results = []
    else:
        error_message = "Не предоставлены данные для классификации (ни файл, ни ручной ввод)."

    # Финальный этап: формирование признаков для модели из all_processed_data_df и предсказание
    if not error_message and not all_processed_data_df.empty: 
        try:
            final_features_for_model = pd.DataFrame(columns=EXPECTED_MODEL_FEATURES)

            for col_name in EXPECTED_MODEL_FEATURES:
                if col_name in all_processed_data_df.columns: 
                    final_features_for_model[col_name] = all_processed_data_df[col_name] 
                else:
                    final_features_for_model[col_name] = np.nan # Будет заполнено заглушкой ниже
            
            # Приведение типов на всякий случай, если что-то пришло как object
            for col_name in final_features_for_model.columns:
                # Попытка преобразовать в числовой тип, если это не объект/строка уже
                if final_features_for_model[col_name].dtype == 'object':
                    final_features_for_model[col_name] = pd.to_numeric(final_features_for_model[col_name], errors='coerce')
                # Если после to_numeric все еще object (значит были неконвертируемые строки), или если изначально числовой, оставляем как есть
                # Главное, чтобы перед fillna были числовые типы там, где ожидаются числа.
            
            # Централизованная обработка NaN с использованием placeholder средних
            for col_name in final_features_for_model.columns:
                if final_features_for_model[col_name].isnull().any():
                    fill_value = PLACEHOLDER_FINAL_FEATURE_MEANS.get(col_name)
                    if fill_value is None: # На случай, если признак не в PLACEHOLDER_FINAL_FEATURE_MEANS (не должно быть)
                        print(f"ВНИМАНИЕ: Для признака {col_name} не найдено значение-заглушка для NaN. Заполняется нулем.")
                        fill_value = 0 
                    final_features_for_model[col_name].fillna(fill_value, inplace=True)
            
            # Убедимся, что колонки в том порядке, который ожидает модель
            final_features_for_model = final_features_for_model[EXPECTED_MODEL_FEATURES]

            predictions_proba = model.predict_proba(final_features_for_model)
            predictions_labels = model.predict(final_features_for_model)
            
            # Используем class_mapping_from_model для имен сегментов
            # Если он не загружен, используем заглушку
            current_class_mapping = class_mapping_from_model if class_mapping_from_model else {
                0: "Класс 0 (Низко)", 
                1: "Класс 1 (Умеренно)",
                2: "Класс 2 (Лояльно)",
                3: "Класс 3 (Высоко)" 
            }
            
            # Попытка получить оригинальные ID клиентов, если они были в исходных данных
            client_ids = ["Клиент_" + str(i+1) for i in range(len(final_features_for_model))] # Заглушка по умолчанию
            if 'Клиент' in all_processed_data_df.columns: 
                client_ids = all_processed_data_df['Клиент'].astype(str).tolist() 
            elif 'client_id' in all_processed_data_df.columns: 
                client_ids = all_processed_data_df['client_id'].astype(str).tolist() 
            
            # Убедимся, что client_ids имеет ту же длину, что и предсказания
            if len(client_ids) != len(predictions_labels):
                client_ids = ["Клиент_" + str(i+1) for i in range(len(predictions_labels))]


            for i, (probas, label) in enumerate(zip(predictions_proba, predictions_labels)):
                results.append({
                    "client_id": client_ids[i],
                    "predicted_segment_id": int(label), # Убедимся, что label это int для JSON
                    "predicted_segment_name": current_class_mapping.get(int(label), f"Неизвестный класс {int(label)}"),
                    "probabilities": {current_class_mapping.get(idx, f"Класс {idx}"): round(p, 4) for idx, p in enumerate(probas)}
                })
            
            if not results:
                 error_message = "Не удалось получить предсказания. Возможно, данные не корректны или отсутствуют."

        except Exception as e:
            error_message = f"Ошибка во время предсказания: {str(e)}"
            import traceback
            traceback.print_exc() # Для детальной ошибки в консоли сервера
            results = []

    return templates.TemplateResponse("partials/classification_result_display.html",
                                    {"request": request, "results": results, "error": error_message, "info_message": info_message, "input_method": input_method})

# Не забудьте добавить этот роутер в основное приложение FastAPI
# в файле src/main.py (или где у вас инициализируется приложение):
# from src.api.routes import ui_routes
# app.include_router(ui_routes.router) 