import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import argparse

# Настройка отображения графиков
plt.style.use('ggplot')
sns.set(style="whitegrid")

def load_and_analyze_structure(file_path, output_dir):
    """
    Загрузка и первичный анализ структуры данных
    """
    print("1. Загрузка и первичный анализ структуры данных")
    
    # Проверка существования файла
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл {file_path} не найден")
        return None
    
    # Информация о размере файла
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # в МБ
    print(f"Размер файла: {file_size:.2f} МБ")
    
    try:
        # Загрузка данных
        print("Загрузка данных...")
        df = pd.read_csv(file_path)
        print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
        
        # Анализ типов данных
        print("\nИнформация о датасете:")
        df.info()
        
        # Анализ пропущенных значений
        missing_values = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame({
            'Кол-во пропущенных': missing_values,
            'Процент пропущенных': missing_percent
        })
        missing_df = missing_df[missing_df['Кол-во пропущенных'] > 0].sort_values('Процент пропущенных', ascending=False)
        
        print("\nПропущенные значения:")
        if len(missing_df) > 0:
            print(missing_df)
            
            # Визуализация пропущенных значений
            if len(missing_df) > 0:
                plt.figure(figsize=(12, 8))
                bars = plt.bar(missing_df.index, missing_df['Процент пропущенных'])
                plt.title('Процент пропущенных значений по колонкам')
                plt.xlabel('Колонки')
                plt.ylabel('Процент пропущенных значений')
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/missing_values.png")
                plt.close()
        else:
            print("Нет пропущенных значений")
            
        # Преобразование дат
        date_columns = ['Дата покупки', 'Дата оформления карты', 'Дата первого чека', 'Дата последнего чека']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Проверка диапазона дат
        if 'Дата покупки' in df.columns:
            print(f"\nПериод данных: с {df['Дата покупки'].min()} по {df['Дата покупки'].max()}")
            print(f"Продолжительность: {(df['Дата покупки'].max() - df['Дата покупки'].min()).days} дней")
            
        return df
        
    except Exception as e:
        print(f"Ошибка при загрузке и анализе файла: {e}")
        return None

def analyze_distributions(df, output_dir):
    """
    Исследование распределений основных переменных
    """
    if df is None or df.empty:
        print("Ошибка: Датасет пуст или не загружен")
        return
    
    print("\n2. Исследование распределений основных переменных")
    
    # Базовая статистика числовых переменных
    print("\nБазовая статистика числовых переменных:")
    print(df.describe().T)
    
    # Анализ распределения категориальных переменных
    if 'Пол' in df.columns:
        plt.figure(figsize=(10, 6))
        gender_counts = df['Пол'].value_counts()
        plt.bar(gender_counts.index, gender_counts.values)
        plt.title('Распределение клиентов по полу')
        plt.ylabel('Количество клиентов')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gender_distribution.png")
        plt.close()
        print(f"\nРаспределение по полу:\n{gender_counts}")
    
    # Топ-10 точек продаж
    if 'Точка продаж' in df.columns:
        top_locations = df['Точка продаж'].value_counts().head(10)
        plt.figure(figsize=(12, 8))
        plt.barh(top_locations.index, top_locations.values)
        plt.title('Топ-10 точек продаж')
        plt.xlabel('Количество покупок')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_locations.png")
        plt.close()
        print(f"\nТоп-10 точек продаж:\n{top_locations}")
    
    # Анализ распределения числовых переменных
    numeric_vars = ['Cумма покупки', 'Начислено бонусов', 'Списано бонусов', 
                   'Средняя сумма покупок', 'Частота, раз/мес']
    
    numeric_vars = [var for var in numeric_vars if var in df.columns]
    
    for var in numeric_vars:
        plt.figure(figsize=(14, 6))
        
        # Гистограмма
        plt.subplot(1, 2, 1)
        sns.histplot(df[var].dropna(), kde=True)
        plt.title(f'Распределение: {var}')
        
        # Boxplot для выявления выбросов
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[var].dropna())
        plt.title(f'Диаграмма размаха: {var}')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/distribution_{var.replace(' ', '_').replace(',', '')}.png")
        plt.close()
        
        # Статистика распределения
        stats = df[var].describe()
        print(f"\nСтатистика для {var}:")
        print(stats)

def aggregate_customer_data(df):
    """
    Агрегация данных на уровне клиента
    """
    if df is None or df.empty:
        print("Ошибка: Датасет пуст или не загружен")
        return None
    
    print("\n3. Агрегация данных на уровне клиента")
    
    if 'Клиент' not in df.columns:
        print("Ошибка: В датасете отсутствует колонка 'Клиент'")
        return None
    
    # Группировка данных по клиентам
    agg_functions = {}
    
    if 'Cумма покупки' in df.columns:
        agg_functions['Cумма покупки'] = ['count', 'sum', 'mean']
    
    if 'Дата покупки' in df.columns:
        agg_functions['Дата покупки'] = ['min', 'max']
    
    if 'Начислено бонусов' in df.columns:
        agg_functions['Начислено бонусов'] = 'sum'
    
    if 'Списано бонусов' in df.columns:
        agg_functions['Списано бонусов'] = 'sum'
    
    if not agg_functions:
        print("Ошибка: Не найдены необходимые колонки для агрегации")
        return None
    
    customer_df = df.groupby('Клиент').agg(agg_functions)
    
    # Переименование колонок
    customer_df.columns = ['_'.join(col).strip() for col in customer_df.columns.values]
    customer_df.reset_index(inplace=True)
    
    # Расчет recency (для последнего дня в датасете)
    if 'Дата покупки_max' in customer_df.columns:
        last_date = df['Дата покупки'].max()
        customer_df['recency'] = (last_date - customer_df['Дата покупки_max']).dt.days
    
    # Переименование колонок для удобства
    rename_dict = {}
    if 'Cумма покупки_count' in customer_df.columns:
        rename_dict['Cумма покупки_count'] = 'frequency'
    if 'Cумма покупки_sum' in customer_df.columns:
        rename_dict['Cумма покупки_sum'] = 'monetary'
    if 'Cумма покупки_mean' in customer_df.columns:
        rename_dict['Cумма покупки_mean'] = 'avg_purchase'
    if 'Начислено бонусов_sum' in customer_df.columns:
        rename_dict['Начислено бонусов_sum'] = 'total_bonus_earned'
    if 'Списано бонусов_sum' in customer_df.columns:
        rename_dict['Списано бонусов_sum'] = 'total_bonus_used'
    
    customer_df = customer_df.rename(columns=rename_dict)
    
    # Базовая статистика по клиентам
    print("\nБазовая статистика по клиентам:")
    print(customer_df.describe())
    
    return customer_df

def perform_rfm_analysis(customer_df, output_dir):
    """
    Анализ RFM-метрик
    """
    if customer_df is None or customer_df.empty:
        print("Ошибка: Агрегированные данные по клиентам отсутствуют")
        return
    
    print("\n4. Анализ RFM-метрик")
    
    required_cols = ['recency', 'frequency', 'monetary']
    missing_cols = [col for col in required_cols if col not in customer_df.columns]
    
    if missing_cols:
        print(f"Ошибка: Отсутствуют необходимые колонки для RFM-анализа: {', '.join(missing_cols)}")
        return
    
    # Создаем квантили для R, F, M
    quantiles = customer_df[required_cols].quantile([0.25, 0.5, 0.75]).to_dict()
    
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
    customer_df['R'] = customer_df['recency'].apply(r_score)
    customer_df['F'] = customer_df['frequency'].apply(lambda x: fm_score(x, 'frequency'))
    customer_df['M'] = customer_df['monetary'].apply(lambda x: fm_score(x, 'monetary'))
    
    # Создание RFM-группы и общего балла
    customer_df['RFM_Group'] = customer_df['R'].astype(str) + customer_df['F'].astype(str) + customer_df['M'].astype(str)
    customer_df['RFM_Score'] = customer_df['R'] + customer_df['F'] + customer_df['M']
    
    # Определение сегментов лояльности
    def loyalty_segment(score):
        if score >= 10:
            return 'Высоколояльные'
        elif score >= 8:
            return 'Умеренно лояльные'
        elif score >= 6:
            return 'Низколояльные'
        elif score >= 4:
            return 'Потенциально лояльные'
        else:
            return 'Группа оттока'
    
    customer_df['Loyalty_Segment'] = customer_df['RFM_Score'].apply(loyalty_segment)
    
    # Визуализация сегментов
    plt.figure(figsize=(12, 6))
    segment_order = ['Высоколояльные', 'Умеренно лояльные', 'Низколояльные', 'Потенциально лояльные', 'Группа оттока']
    segment_counts = customer_df['Loyalty_Segment'].value_counts().reindex(segment_order)
    plt.bar(segment_counts.index, segment_counts.values)
    plt.title('Распределение клиентов по сегментам лояльности')
    plt.xlabel('Сегмент лояльности')
    plt.ylabel('Количество клиентов')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loyalty_segments.png")
    plt.close()
    
    # Статистика по сегментам
    segment_stats = customer_df.groupby('Loyalty_Segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'Клиент': 'count'
    }).rename(columns={'Клиент': 'Count'}).reset_index()
    
    print("\nСтатистика по сегментам лояльности:")
    print(segment_stats)
    
    # Сохранение результатов в CSV
    customer_df.to_csv(f"{output_dir}/customer_rfm_segments.csv", index=False)
    print(f"Результаты RFM-анализа сохранены в {output_dir}/customer_rfm_segments.csv")

def analyze_relationships(customer_df, output_dir):
    """
    Исследование взаимосвязей между переменными
    """
    if customer_df is None or customer_df.empty:
        print("Ошибка: Агрегированные данные по клиентам отсутствуют")
        return
    
    print("\n5. Исследование взаимосвязей между переменными")
    
    # Корреляция между числовыми переменными
    corr_cols = ['recency', 'frequency', 'monetary', 'avg_purchase', 
                'total_bonus_earned', 'total_bonus_used']
    corr_cols = [col for col in corr_cols if col in customer_df.columns]
    
    if len(corr_cols) > 1:
        correlation = customer_df[corr_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Корреляция между показателями лояльности')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_matrix.png")
        plt.close()
        
        print("\nМатрица корреляции:")
        print(correlation)
    
    # Взаимосвязь между частотой и суммой покупок
    if 'frequency' in customer_df.columns and 'monetary' in customer_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='frequency', y='monetary', data=customer_df)
        plt.title('Взаимосвязь между частотой и суммой покупок')
        plt.xlabel('Количество покупок')
        plt.ylabel('Общая сумма покупок, руб.')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/frequency_monetary_relationship.png")
        plt.close()
    
    # Взаимосвязь между суммой покупок и начисленными бонусами
    if 'monetary' in customer_df.columns and 'total_bonus_earned' in customer_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='monetary', y='total_bonus_earned', data=customer_df)
        plt.title('Взаимосвязь между суммой покупок и начисленными бонусами')
        plt.xlabel('Общая сумма покупок, руб.')
        plt.ylabel('Начислено бонусов, руб.')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/monetary_bonus_relationship.png")
        plt.close()
    
    # Средние показатели по сегментам лояльности
    if 'Loyalty_Segment' in customer_df.columns:
        segment_metrics = ['recency', 'frequency', 'monetary', 'avg_purchase']
        segment_metrics = [metric for metric in segment_metrics if metric in customer_df.columns]
        
        if segment_metrics:
            segment_order = ['Высоколояльные', 'Умеренно лояльные', 'Низколояльные', 'Потенциально лояльные', 'Группа оттока']
            
            for metric in segment_metrics:
                plt.figure(figsize=(12, 6))
                sns.barplot(x='Loyalty_Segment', y=metric, data=customer_df, order=segment_order)
                plt.title(f'Средний {metric} по сегментам лояльности')
                plt.xlabel('Сегмент лояльности')
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/segment_{metric}.png")
                plt.close()

def perform_temporal_analysis(df, output_dir):
    """
    Временной анализ
    """
    if df is None or df.empty:
        print("Ошибка: Датасет пуст или не загружен")
        return
    
    print("\n6. Временной анализ")
    
    if 'Дата покупки' not in df.columns:
        print("Ошибка: В датасете отсутствует колонка 'Дата покупки'")
        return
    
    # Проверка, что колонка с датой имеет правильный тип
    if not pd.api.types.is_datetime64_any_dtype(df['Дата покупки']):
        df['Дата покупки'] = pd.to_datetime(df['Дата покупки'], errors='coerce')
    
    # Анализ покупок по дням недели
    df['day_of_week'] = df['Дата покупки'].dt.day_name()
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_of_week'].value_counts().reindex(day_order)
    
    plt.figure(figsize=(12, 6))
    plt.bar(day_counts.index, day_counts.values)
    plt.title('Распределение покупок по дням недели')
    plt.xlabel('День недели')
    plt.ylabel('Количество покупок')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/day_of_week_distribution.png")
    plt.close()
    
    print("\nРаспределение покупок по дням недели:")
    print(day_counts)
    
    # Анализ покупок по дням месяца
    df['day_of_month'] = df['Дата покупки'].dt.day
    
    day_of_month_counts = df.groupby('day_of_month').size()
    
    plt.figure(figsize=(14, 6))
    plt.bar(day_of_month_counts.index, day_of_month_counts.values)
    plt.title('Распределение покупок по дням месяца')
    plt.xlabel('День месяца')
    plt.ylabel('Количество покупок')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/day_of_month_distribution.png")
    plt.close()
    
    # Анализ покупок по времени (если есть временная часть в дате)
    if hasattr(df['Дата покупки'].dt, 'hour') and df['Дата покупки'].dt.hour.nunique() > 1:
        df['hour_of_day'] = df['Дата покупки'].dt.hour
        
        hour_counts = df.groupby('hour_of_day').size()
        
        plt.figure(figsize=(14, 6))
        plt.bar(hour_counts.index, hour_counts.values)
        plt.title('Распределение покупок по часам дня')
        plt.xlabel('Час дня')
        plt.ylabel('Количество покупок')
        plt.xticks(range(0, 24))
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hour_of_day_distribution.png")
        plt.close()
        
        print("\nРаспределение покупок по часам дня:")
        print(hour_counts)

def analyze_product_categories(df, output_dir):
    """
    Анализ товарных категорий
    """
    if df is None or df.empty:
        print("Ошибка: Датасет пуст или не загружен")
        return
    
    print("\n7. Анализ товарных категорий")
    
    if 'Название товара' not in df.columns:
        print("Ошибка: В датасете отсутствует колонка 'Название товара'")
        return
    
    # Топ-10 самых популярных товаров
    top_products = df['Название товара'].value_counts().head(10)
    
    plt.figure(figsize=(12, 8))
    plt.barh(top_products.index, top_products.values)
    plt.title('Топ-10 самых популярных товаров')
    plt.xlabel('Количество покупок')
    plt.ylabel('Название товара')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_products.png")
    plt.close()
    
    print("\nТоп-10 самых популярных товаров:")
    print(top_products)
    
    # Средний чек по товарным категориям
    if 'Cумма покупки' in df.columns:
        product_avg = df.groupby('Название товара')['Cумма покупки'].mean().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 8))
        plt.barh(product_avg.index, product_avg.values)
        plt.title('Топ-10 товаров по среднему чеку')
        plt.xlabel('Средний чек, руб.')
        plt.ylabel('Название товара')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_products_by_avg_check.png")
        plt.close()
        
        print("\nТоп-10 товаров по среднему чеку:")
        print(product_avg)

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Exploratory Data Analysis for Acoola Customer Loyalty')
    parser.add_argument('--data_path', type=str, default='../dataset/Concept202408.csv', help='Path to the CSV data file')
    parser.add_argument('--output_dir', type=str, default='../output', help='Directory to save output plots and data')
    args = parser.parse_args()
    
    # Создание директории для вывода, если она не существует
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Создана директория для вывода: {args.output_dir}")
    
    print(f"Запуск исследовательского анализа данных для файла: {args.data_path}")
    print(f"Результаты будут сохранены в: {args.output_dir}")
    
    # Загрузка и первичный анализ структуры данных
    df = load_and_analyze_structure(args.data_path, args.output_dir)
    
    if df is not None:
        # Исследование распределений основных переменных
        analyze_distributions(df, args.output_dir)
        
        # Агрегация данных на уровне клиента
        customer_df = aggregate_customer_data(df)
        
        if customer_df is not None:
            # Анализ RFM-метрик
            perform_rfm_analysis(customer_df, args.output_dir)
            
            # Исследование взаимосвязей между переменными
            analyze_relationships(customer_df, args.output_dir)
        
        # Временной анализ
        perform_temporal_analysis(df, args.output_dir)
        
        # Анализ товарных категорий
        analyze_product_categories(df, args.output_dir)
        
        print("\nИсследовательский анализ данных успешно завершен!")
        print(f"Результаты сохранены в директории: {args.output_dir}")
    else:
        print("Ошибка при загрузке данных. Анализ невозможен.")

if __name__ == "__main__":
    main() 