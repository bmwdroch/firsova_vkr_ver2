import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from scipy import stats
import os
import argparse

def preprocess_data(df, output_dir=None):
    """
    Комплексная предобработка данных для моделирования
    
    Аргументы:
        df (pandas.DataFrame): Исходный датасет
        output_dir (str, optional): Директория для сохранения графиков и отчетов
        
    Возвращает:
        pandas.DataFrame: Предобработанный датасет
    """
    if df is None or df.empty:
        print("Ошибка: Датасет пуст или не загружен")
        return None
    
    print("Начало предобработки данных")
    
    # Шаг 1: Обработка пропущенных значений
    df = handle_missing_values(df)
    
    # Шаг 2: Обработка выбросов
    df = handle_outliers(df)
    
    # Шаг 3: Нормализация числовых признаков
    df = normalize_numeric_features(df)
    
    # Шаг 4: Кодирование категориальных переменных
    df = encode_categorical_features(df)
    
    # Шаг 5: Создание агрегированных данных на уровне клиента
    customer_df = create_aggregated_features(df)
    
    # Шаг 6: Подготовка итогового датасета
    final_df = prepare_final_dataset(customer_df)
    
    # Сохранение предобработанных данных
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if output_dir:
        final_df.to_csv(f"{output_dir}/preprocessed_data.csv", index=False)
        print(f"Предобработанные данные сохранены в {output_dir}/preprocessed_data.csv")
    
    return final_df

def handle_missing_values(df):
    """
    Обработка пропущенных значений в датасете
    
    Аргументы:
        df (pandas.DataFrame): Исходный датасет
        
    Возвращает:
        pandas.DataFrame: Датасет с обработанными пропущенными значениями
    """
    print("Обработка пропущенных значений...")
    
    # Копирование датасета для избежания изменений исходных данных
    df_clean = df.copy()
    
    # Подсчет пропущенных значений до обработки
    missing_before = df_clean.isnull().sum().sum()
    print(f"Количество пропущенных значений до обработки: {missing_before}")
    
    # Даты: заполнение медианой или первой доступной датой для клиента
    date_columns = ['Дата оформления карты', 'Дата первого чека', 'Дата последнего чека']
    for col in date_columns:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            # Для каждого клиента заполняем пропуски медианой его дат
            if 'Клиент' in df_clean.columns:
                df_clean[col] = df_clean.groupby('Клиент')[col].transform(
                    lambda x: x.fillna(x.median() if not x.median() is pd.NaT else pd.NaT)
                )
            
            # Оставшиеся пропуски заполняем общей медианой
            if df_clean[col].isnull().sum() > 0:
                median_date = df_clean[col].median()
                if not pd.isna(median_date):
                    df_clean[col] = df_clean[col].fillna(median_date)
    
    # Пол: заполнение наиболее частым значением
    if 'Пол' in df_clean.columns and df_clean['Пол'].isnull().sum() > 0:
        most_common_gender = df_clean['Пол'].mode()[0]
        df_clean['Пол'] = df_clean['Пол'].fillna(most_common_gender)
    
    # Числовые данные: заполнение медианой по группам (если это уместно) или общей медианой
    numeric_cols = ['Cумма покупки', 'Начислено бонусов', 'Списано бонусов', 
                   'Средняя сумма покупок', 'Частота, раз/мес', 'Баланс накопленный', 
                   'Баланс подарочный', 'Покупок, в днях']
    
    numeric_cols = [col for col in numeric_cols if col in df_clean.columns]
    
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            # Для каждого клиента заполняем пропуски медианой его значений
            if 'Клиент' in df_clean.columns:
                df_clean[col] = df_clean.groupby('Клиент')[col].transform(
                    lambda x: x.fillna(x.median())
                )
            
            # Оставшиеся пропуски заполняем общей медианой
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Категориальные данные: заполнение наиболее частым значением или специальной категорией "Неизвестно"
    categorical_cols = ['Точка продаж', 'Точка продаж где оформлена карта', 'Название товара']
    categorical_cols = [col for col in categorical_cols if col in df_clean.columns]
    
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            # Если пропусков мало, заполняем наиболее частым значением
            if df_clean[col].isnull().sum() / len(df_clean) < 0.05:
                most_common = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(most_common)
            else:
                # Если пропусков много, создаем категорию "Неизвестно"
                df_clean[col] = df_clean[col].fillna("Неизвестно")
    
    # Подсчет пропущенных значений после обработки
    missing_after = df_clean.isnull().sum().sum()
    print(f"Количество пропущенных значений после обработки: {missing_after}")
    print(f"Обработано пропущенных значений: {missing_before - missing_after}")
    
    return df_clean

def handle_outliers(df):
    """
    Обработка выбросов в числовых переменных методом IQR (межквартильного размаха)
    
    Аргументы:
        df (pandas.DataFrame): Исходный датасет
        
    Возвращает:
        pandas.DataFrame: Датасет с обработанными выбросами
    """
    print("Обработка выбросов...")
    
    # Копирование датасета
    df_clean = df.copy()
    
    # Список числовых переменных для обработки выбросов
    numeric_vars = ['Cумма покупки', 'Начислено бонусов', 'Списано бонусов', 
                    'Средняя сумма покупок', 'Частота, раз/мес', 'Баланс накопленный', 
                    'Баланс подарочный']
    
    # Оставляем только те колонки, которые существуют в датасете
    numeric_vars = [var for var in numeric_vars if var in df_clean.columns]
    
    # Обработка выбросов для каждой переменной
    for var in numeric_vars:
        # Расчет квартилей
        Q1 = df_clean[var].quantile(0.25)
        Q3 = df_clean[var].quantile(0.75)
        IQR = Q3 - Q1
        
        # Определение границ для выбросов
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Количество выбросов до обработки
        outliers_count = ((df_clean[var] < lower_bound) | (df_clean[var] > upper_bound)).sum()
        
        # Обработка выбросов с помощью винсоризации (ограничение значениями)
        if outliers_count > 0:
            print(f"Переменная '{var}': обнаружено {outliers_count} выбросов ({outliers_count/len(df_clean)*100:.2f}%)")
            
            # Для верхней границы применяем винсоризацию (замена на порог)
            df_clean.loc[df_clean[var] > upper_bound, var] = upper_bound
            
            # Для нижней границы тоже применяем винсоризацию, если необходимо
            # Например, для сумм покупок отрицательные значения могут быть возвратами
            if lower_bound > 0:  # Только если нижняя граница положительная
                df_clean.loc[df_clean[var] < lower_bound, var] = lower_bound
    
    return df_clean

def normalize_numeric_features(df):
    """
    Нормализация/стандартизация числовых признаков
    
    Аргументы:
        df (pandas.DataFrame): Исходный датасет
        
    Возвращает:
        pandas.DataFrame: Датасет с нормализованными числовыми признаками
    """
    print("Нормализация числовых признаков...")
    
    # Копирование датасета
    df_clean = df.copy()
    
    # Список числовых переменных для нормализации
    monetary_vars = ['Cумма покупки', 'Начислено бонусов', 'Списано бонусов', 
                     'Средняя сумма покупок', 'Баланс накопленный', 'Баланс подарочный']
    
    frequency_vars = ['Частота, раз/мес', 'Покупок, в днях']
    
    # Оставляем только те колонки, которые существуют в датасете
    monetary_vars = [var for var in monetary_vars if var in df_clean.columns]
    frequency_vars = [var for var in frequency_vars if var in df_clean.columns]
    
    # Для денежных переменных применяем MinMaxScaler (масштабирование к диапазону [0, 1])
    if monetary_vars:
        scaler = MinMaxScaler()
        df_clean[monetary_vars] = scaler.fit_transform(df_clean[monetary_vars])
        print(f"Применено масштабирование к диапазону [0, 1] для переменных: {', '.join(monetary_vars)}")
    
    # Для частотных переменных применяем StandardScaler (стандартизация)
    if frequency_vars:
        scaler = StandardScaler()
        df_clean[frequency_vars] = scaler.fit_transform(df_clean[frequency_vars])
        print(f"Применена стандартизация для переменных: {', '.join(frequency_vars)}")
    
    return df_clean

def encode_categorical_features(df):
    """
    Преобразование категориальных переменных
    
    Аргументы:
        df (pandas.DataFrame): Исходный датасет
        
    Возвращает:
        pandas.DataFrame: Датасет с преобразованными категориальными переменными
    """
    print("Кодирование категориальных переменных...")
    
    # Копирование датасета
    df_clean = df.copy()
    
    # Пол: label encoding (0/1)
    if 'Пол' in df_clean.columns:
        le = LabelEncoder()
        df_clean['Пол_encoded'] = le.fit_transform(df_clean['Пол'])
        # Сохраняем маппинг для интерпретации
        gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Кодирование переменной 'Пол': {gender_mapping}")
    
    # Точка продаж: one-hot encoding для часто встречающихся значений
    if 'Точка продаж' in df_clean.columns:
        # Выбираем топ-N наиболее частых точек продаж
        N = 10  # Можно настроить в зависимости от распределения
        top_locations = df_clean['Точка продаж'].value_counts().nlargest(N).index
        
        # Создаем новый признак для определения, относится ли точка к топ-N
        df_clean['top_location'] = df_clean['Точка продаж'].apply(lambda x: x if x in top_locations else 'Другое')
        
        # Применяем one-hot encoding
        ohe = pd.get_dummies(df_clean['top_location'], prefix='location')
        df_clean = pd.concat([df_clean, ohe], axis=1)
        
        print(f"Созданы one-hot encoding признаки для {N} наиболее частых точек продаж")
    
    # Название товара: создаем агрегированные категории и one-hot encoding
    if 'Название товара' in df_clean.columns:
        # Предполагаем, что мы можем определить основные категории товаров
        # Например, группируя по ключевым словам в названии
        
        # Функция для определения категории товара
        def categorize_product(product_name):
            if pd.isna(product_name):
                return 'Неизвестно'
            
            product_name = product_name.lower()
            
            # Пример категоризации для детской одежды Acoola
            if 'куртка' in product_name or 'пальто' in product_name:
                return 'Верхняя одежда'
            elif 'футболка' in product_name or 'майка' in product_name:
                return 'Футболки'
            elif 'брюки' in product_name or 'джинсы' in product_name:
                return 'Брюки'
            elif 'платье' in product_name:
                return 'Платья'
            elif 'юбка' in product_name:
                return 'Юбки'
            elif 'рубашка' in product_name or 'блуза' in product_name:
                return 'Рубашки'
            elif 'костюм' in product_name:
                return 'Костюмы'
            elif 'аксессуар' in product_name:
                return 'Аксессуары'
            else:
                return 'Другое'
        
        # Применяем категоризацию
        df_clean['product_category'] = df_clean['Название товара'].apply(categorize_product)
        
        # Применяем one-hot encoding к категориям товаров
        ohe = pd.get_dummies(df_clean['product_category'], prefix='product')
        df_clean = pd.concat([df_clean, ohe], axis=1)
        
        print(f"Созданы one-hot encoding признаки для категорий товаров")
    
    return df_clean

def create_aggregated_features(df):
    """
    Создание агрегированных данных на уровне клиента
    
    Аргументы:
        df (pandas.DataFrame): Исходный датасет с транзакциями
        
    Возвращает:
        pandas.DataFrame: Агрегированный датасет на уровне клиента
    """
    print("Создание агрегированных данных на уровне клиента...")
    
    # Проверка наличия ключевой колонки
    if 'Клиент' not in df.columns:
        print("Ошибка: В датасете отсутствует колонка 'Клиент'")
        return df
    
    # Группировка данных по клиентам
    agg_functions = {}
    
    # Базовые метрики
    if 'Cумма покупки' in df.columns:
        agg_functions['Cумма покупки'] = ['count', 'sum', 'mean', 'std']
    
    if 'Дата покупки' in df.columns:
        agg_functions['Дата покупки'] = ['min', 'max']
    
    if 'Начислено бонусов' in df.columns:
        agg_functions['Начислено бонусов'] = ['sum', 'mean']
    
    if 'Списано бонусов' in df.columns:
        agg_functions['Списано бонусов'] = ['sum', 'mean']
    
    if 'Пол_encoded' in df.columns:
        # Для кодированного пола берем моду (наиболее частое значение)
        agg_functions['Пол_encoded'] = pd.Series.mode
    
    # One-hot encoded переменные (точки продаж, категории товаров)
    location_cols = [col for col in df.columns if col.startswith('location_')]
    product_cols = [col for col in df.columns if col.startswith('product_')]
    
    for col in location_cols + product_cols:
        agg_functions[col] = 'mean'  # Доля покупок в данной точке или категории
    
    # Проверка наличия функций агрегации
    if not agg_functions:
        print("Ошибка: Не найдены необходимые колонки для агрегации")
        return df
    
    # Группировка данных
    customer_df = df.groupby('Клиент').agg(agg_functions)
    
    # Переименование колонок
    customer_df.columns = ['_'.join(col).strip() for col in customer_df.columns.values]
    customer_df.reset_index(inplace=True)
    
    # Расчет recency (для последнего дня в датасете)
    if 'Дата покупки_max' in customer_df.columns:
        last_date = df['Дата покупки'].max()
        customer_df['recency'] = (last_date - customer_df['Дата покупки_max']).dt.days
    
    # Расчет дополнительных агрегированных признаков
    
    # Процент использования бонусов
    if 'Начислено бонусов_sum' in customer_df.columns and 'Списано бонусов_sum' in customer_df.columns:
        customer_df['bonus_usage_ratio'] = customer_df['Списано бонусов_sum'] / customer_df['Начислено бонусов_sum'].replace(0, 1)
    
    # Период активности клиента (в днях)
    if 'Дата покупки_min' in customer_df.columns and 'Дата покупки_max' in customer_df.columns:
        customer_df['activity_period'] = (customer_df['Дата покупки_max'] - customer_df['Дата покупки_min']).dt.days
        customer_df['activity_period'] = customer_df['activity_period'].replace(0, 1)  # Избегаем деления на 0
    
    # Средняя частота покупок (покупок в день за период активности)
    if 'Cумма покупки_count' in customer_df.columns and 'activity_period' in customer_df.columns:
        customer_df['purchase_frequency'] = customer_df['Cумма покупки_count'] / customer_df['activity_period']
    
    # Переименование колонок для удобства
    rename_dict = {}
    if 'Cумма покупки_count' in customer_df.columns:
        rename_dict['Cумма покупки_count'] = 'frequency'
    if 'Cумма покупки_sum' in customer_df.columns:
        rename_dict['Cумма покупки_sum'] = 'monetary'
    if 'Cумма покупки_mean' in customer_df.columns:
        rename_dict['Cумма покупки_mean'] = 'avg_purchase'
    
    if rename_dict:
        customer_df.rename(columns=rename_dict, inplace=True)
    
    print(f"Создан агрегированный датасет с {len(customer_df)} клиентами и {len(customer_df.columns)} признаками")
    
    return customer_df

def prepare_final_dataset(customer_df):
    """
    Подготовка итогового датасета для моделирования, включая RFM-анализ
    
    Аргументы:
        customer_df (pandas.DataFrame): Агрегированный датасет на уровне клиента
        
    Возвращает:
        pandas.DataFrame: Финальный датасет, готовый для моделирования
    """
    print("Подготовка итогового датасета...")
    
    # Копирование датасета
    final_df = customer_df.copy()
    
    # Удаление ненужных или избыточных признаков
    cols_to_drop = []
    
    # Удаление колонок с большим количеством пропущенных значений (> 30%)
    missing_percent = final_df.isnull().sum() / len(final_df) * 100
    high_missing = missing_percent[missing_percent > 30].index.tolist()
    cols_to_drop.extend(high_missing)
    
    # Проверка наличия необходимых RFM-переменных
    required_rfm = ['recency', 'frequency', 'monetary']
    missing_rfm = [col for col in required_rfm if col not in final_df.columns]
    
    # Если все RFM-метрики доступны, выполняем RFM-анализ
    if not missing_rfm:
        # Создаем квантили для R, F, M
        quantiles = final_df[required_rfm].quantile([0.25, 0.5, 0.75]).to_dict()
        
        # Функции для присвоения RFM-оценок
        def r_score(x):
            if x <= quantiles['recency'][0.25]:
                return 4
            elif x <= quantiles['recency'][0.5]:
                return 3
            elif x <= quantiles['recency'][0.75]:
                return 2
            else:
                return 1
        
        def fm_score(x, metric):
            if x <= quantiles[metric][0.25]:
                return 1
            elif x <= quantiles[metric][0.5]:
                return 2
            elif x <= quantiles[metric][0.75]:
                return 3
            else:
                return 4
        
        # Присвоение RFM-оценок
        final_df['R'] = final_df['recency'].apply(r_score)
        final_df['F'] = final_df['frequency'].apply(lambda x: fm_score(x, 'frequency'))
        final_df['M'] = final_df['monetary'].apply(lambda x: fm_score(x, 'monetary'))
        
        # Создание RFM-группы и общего балла
        final_df['RFM_Group'] = final_df['R'].astype(str) + final_df['F'].astype(str) + final_df['M'].astype(str)
        final_df['RFM_Score'] = final_df['R'] + final_df['F'] + final_df['M']
        
        # Определение сегмента лояльности на основе RFM-оценки
        def loyalty_segment(score):
            if score >= 9:
                return 'Высоколояльные'
            elif score >= 7:
                return 'Лояльные'
            elif score >= 5:
                return 'Умеренно лояльные'
            elif score >= 3:
                return 'Низколояльные'
            else:
                return 'Отток'
        
        final_df['loyalty_segment'] = final_df['RFM_Score'].apply(loyalty_segment)
        
        print("RFM-анализ выполнен, созданы сегменты лояльности клиентов")
    else:
        print(f"ВНИМАНИЕ: Невозможно выполнить RFM-анализ. Отсутствуют колонки: {', '.join(missing_rfm)}")
    
    # Удаление выбранных колонок (если необходимо)
    if cols_to_drop:
        final_df.drop(columns=cols_to_drop, inplace=True)
        print(f"Удалены избыточные колонки: {', '.join(cols_to_drop)}")
    
    # Проверка и удаление оставшихся пропущенных значений
    if final_df.isnull().sum().sum() > 0:
        final_df.dropna(inplace=True)
        print(f"Удалены строки с пропущенными значениями. Итоговый размер датасета: {len(final_df)} клиентов")
    
    print(f"Подготовлен итоговый датасет для моделирования, содержащий {len(final_df)} клиентов и {len(final_df.columns)} признаков")
    
    return final_df

def main():
    """
    Основная функция для запуска предобработки данных
    """
    parser = argparse.ArgumentParser(description='Предобработка данных для моделирования')
    parser.add_argument('--input', type=str, required=True, help='Путь к исходному CSV-файлу')
    parser.add_argument('--output', type=str, default='../output', help='Директория для сохранения результатов')
    args = parser.parse_args()
    
    # Проверка существования файла
    if not os.path.exists(args.input):
        print(f"Ошибка: Файл {args.input} не найден")
        return
    
    # Создание выходной директории, если она не существует
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Загрузка данных
    print(f"Загрузка данных из {args.input}...")
    df = pd.read_csv(args.input)
    
    # Предобработка данных
    preprocessed_df = preprocess_data(df, args.output)
    
    if preprocessed_df is not None:
        print("Предобработка данных завершена успешно.")
    else:
        print("Ошибка при предобработке данных.")

if __name__ == "__main__":
    main() 