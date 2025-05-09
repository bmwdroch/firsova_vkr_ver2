import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Указываем путь к файлу данных
FILE_PATH = '../dataset/Concept202408.csv'

def analyze_dataset(file_path):
    """
    Анализирует CSV файл и выводит основную информацию о нем
    """
    print(f"Анализ датасета: {file_path}")
    
    # Проверка существования файла
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл {file_path} не найден")
        return
    
    # Информация о размере файла
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # в МБ
    print(f"Размер файла: {file_size:.2f} МБ")
    
    try:
        # Чтение данных (можно ограничить количество строк для быстрого анализа)
        print("Загрузка данных...")
        df = pd.read_csv(file_path)
        
        # Основная информация о датасете
        print(f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
        
        # Информация о колонках
        print("\nНазвания колонок:")
        for col in df.columns:
            print(f"  - {col}")
        
        # Типы данных
        print("\nТипы данных:")
        for col, dtype in df.dtypes.items():
            print(f"  - {col}: {dtype}")
        
        # Пропущенные значения
        print("\nПропущенные значения:")
        missing_values = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame({
            'Кол-во пропущенных': missing_values,
            'Процент пропущенных': missing_percent
        })
        missing_df = missing_df[missing_df['Кол-во пропущенных'] > 0].sort_values('Процент пропущенных', ascending=False)
        
        if len(missing_df) > 0:
            for col, count in missing_df['Кол-во пропущенных'].items():
                percent = missing_df.loc[col, 'Процент пропущенных']
                print(f"  - {col}: {count} ({percent:.2f}%)")
        else:
            print("  Нет пропущенных значений")
        
        # Дубликаты
        duplicates = df.duplicated().sum()
        print(f"\nКоличество дубликатов: {duplicates} ({duplicates/len(df)*100:.2f}%)")
        
        # Статистика по числовым полям
        print("\nСтатистика по числовым полям:")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().T
            for col in stats.index:
                print(f"  - {col}:")
                print(f"    Среднее: {stats.loc[col, 'mean']:.2f}")
                print(f"    Мин: {stats.loc[col, 'min']:.2f}")
                print(f"    Макс: {stats.loc[col, 'max']:.2f}")
                print(f"    Медиана: {stats.loc[col, '50%']:.2f}")
        
        # Анализ категориальных переменных
        print("\nАнализ категориальных переменных:")
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                unique_values = df[col].nunique()
                print(f"  - {col}: {unique_values} уникальных значений")
                
                # Если уникальных значений мало, покажем их распределение
                if unique_values <= 10:
                    value_counts = df[col].value_counts()
                    for val, count in value_counts.items():
                        print(f"    - {val}: {count} ({count/len(df)*100:.2f}%)")
        
        # Анализ дат, если они есть
        print("\nПоиск дат в данных:")
        date_cols = []
        for col in df.columns:
            if 'дата' in col.lower() or 'date' in col.lower():
                date_cols.append(col)
                
        if len(date_cols) > 0:
            for col in date_cols:
                print(f"  - {col}:")
                try:
                    df[col] = pd.to_datetime(df[col])
                    min_date = df[col].min()
                    max_date = df[col].max()
                    print(f"    Временной диапазон: с {min_date} по {max_date}")
                    print(f"    Период данных: {(max_date - min_date).days} дней")
                except:
                    print("    Не удалось преобразовать в формат даты")
        
        # Анализ данных по клиентам, если есть такая колонка
        if 'Клиент' in df.columns:
            print("\nАнализ данных по клиентам:")
            unique_customers = df['Клиент'].nunique()
            print(f"  Уникальных клиентов: {unique_customers}")
            
            # Количество транзакций на клиента
            transactions_per_customer = df.groupby('Клиент').size()
            print(f"  Среднее количество транзакций на клиента: {transactions_per_customer.mean():.2f}")
            print(f"  Медиана количества транзакций на клиента: {transactions_per_customer.median():.2f}")
            print(f"  Максимальное количество транзакций у клиента: {transactions_per_customer.max()}")
            
            # Если есть сумма покупки, посчитаем средний чек
            if 'Cумма покупки' in df.columns:
                avg_purchase = df.groupby('Клиент')['Cумма покупки'].mean()
                print(f"  Средний чек по всем клиентам: {avg_purchase.mean():.2f}")
                print(f"  Медиана среднего чека: {avg_purchase.median():.2f}")
                
                # Общая сумма покупок на клиента
                total_purchase = df.groupby('Клиент')['Cумма покупки'].sum()
                print(f"  Средняя общая сумма покупок на клиента: {total_purchase.mean():.2f}")
                print(f"  Медиана общей суммы покупок: {total_purchase.median():.2f}")
                print(f"  Максимальная сумма покупок у клиента: {total_purchase.max():.2f}")
        
        print("\nАнализ завершен успешно!")
        
    except Exception as e:
        print(f"Ошибка при анализе файла: {e}")

if __name__ == "__main__":
    analyze_dataset(FILE_PATH) 