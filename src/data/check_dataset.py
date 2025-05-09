import pandas as pd
import sys
import os

def check_csv_file(filepath):
    """
    Проверяет структуру CSV файла, выводя информацию о колонках и первых строках
    """
    print(f"Проверка файла: {filepath}")
    
    # Проверка существования файла
    if not os.path.exists(filepath):
        print(f"Ошибка: Файл {filepath} не найден")
        return
    
    # Информация о размере файла
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # в МБ
    print(f"Размер файла: {file_size:.2f} МБ")
    
    try:
        # Чтение первых строк для определения структуры
        df_sample = pd.read_csv(filepath, nrows=5)
        
        # Информация о колонках
        print(f"\nКоличество колонок: {len(df_sample.columns)}")
        print("Названия колонок:")
        for col in df_sample.columns:
            print(f"  - {col}")
        
        # Вывод первых 5 строк
        print("\nПервые 5 строк:")
        print(df_sample.to_string())
        
        # Определение типов данных
        print("\nТипы данных:")
        for col, dtype in df_sample.dtypes.items():
            print(f"  - {col}: {dtype}")
        
        # Проверка количества строк (подсчет может занять время для больших файлов)
        print("\nПодсчет общего количества строк (может занять время)...")
        total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1  # Вычитаем строку заголовка
        print(f"Общее количество строк: {total_rows}")
        
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        
if __name__ == "__main__":
    filepath = "../dataset/Concept202408.csv"
    check_csv_file(filepath) 