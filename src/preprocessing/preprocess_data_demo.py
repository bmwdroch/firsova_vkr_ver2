import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.data_preprocessing import preprocess_data

def main():
    """
    Демонстрационный запуск предобработки данных с визуализацией результатов
    """
    # Пути к файлам и директориям
    input_file = "../dataset/Concept202408.csv"
    # Альтернативный путь, если файл в текущей директории
    if not os.path.exists(input_file):
        input_file = "./dataset/Concept202408.csv"
        if not os.path.exists(input_file):
            input_file = "Concept202408.csv"
    
    output_dir = "../output"
    if not os.path.exists(output_dir):
        output_dir = "./output"
    
    # Проверка существования файла
    if not os.path.exists(input_file):
        print(f"Ошибка: Файл {input_file} не найден")
        print(f"Текущая директория: {os.getcwd()}")
        print("Содержимое директории:")
        for item in os.listdir('.'):
            print(f"  - {item}")
        return
    
    # Создание выходной директории, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Загрузка данных
    print(f"Загрузка данных из {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"Датасет успешно загружен. Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return
    
    # Демонстрация распределения некоторых переменных до предобработки
    try:
        demo_before_preprocessing(df, output_dir)
    except Exception as e:
        print(f"Ошибка при визуализации данных до предобработки: {e}")
    
    # Выполнение предобработки данных
    preprocessed_df = preprocess_data(df, output_dir)
    
    if preprocessed_df is not None:
        print("\nПредобработка данных завершена успешно.")
        
        # Демонстрация распределения лояльности клиентов после предобработки
        try:
            demo_after_preprocessing(preprocessed_df, output_dir)
        except Exception as e:
            print(f"Ошибка при визуализации данных после предобработки: {e}")
    else:
        print("Ошибка при предобработке данных.")

def demo_before_preprocessing(df, output_dir):
    """
    Визуализация данных до предобработки
    """
    print("\nВизуализация данных до предобработки...")
    
    # Проверка наличия числовых переменных
    numeric_vars = ['Cумма покупки', 'Начислено бонусов', 'Списано бонусов']
    available_vars = [var for var in numeric_vars if var in df.columns]
    
    if not available_vars:
        print("Не найдены числовые переменные для визуализации")
        return
    
    # Создание директории для визуализаций
    vis_dir = os.path.join(output_dir, "visualization")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Визуализация распределений до предобработки
    for var in available_vars:
        plt.figure(figsize=(12, 6))
        
        # Гистограмма
        plt.subplot(1, 2, 1)
        sns.histplot(df[var].dropna(), kde=True)
        plt.title(f'Распределение до обработки: {var}')
        
        # Boxplot для выявления выбросов
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[var].dropna())
        plt.title(f'Диаграмма размаха до обработки: {var}')
        
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/before_{var.replace(' ', '_').replace(',', '')}.png")
        plt.close()
        
        print(f"Визуализация для '{var}' сохранена")

def demo_after_preprocessing(preprocessed_df, output_dir):
    """
    Визуализация данных после предобработки
    """
    print("\nВизуализация данных после предобработки...")
    
    # Создание директории для визуализаций
    vis_dir = os.path.join(output_dir, "visualization")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Визуализация распределения RFM-сегментов
    if 'RFM_Score' in preprocessed_df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='RFM_Score', data=preprocessed_df, palette='viridis')
        plt.title('Распределение клиентов по RFM-баллам')
        plt.xlabel('RFM-балл')
        plt.ylabel('Количество клиентов')
        plt.savefig(f"{vis_dir}/rfm_score_distribution.png")
        plt.close()
        print("Визуализация RFM-баллов сохранена")
    
    # Визуализация сегментов лояльности
    if 'loyalty_segment' in preprocessed_df.columns:
        plt.figure(figsize=(12, 6))
        loyalty_counts = preprocessed_df['loyalty_segment'].value_counts().sort_index()
        
        # Создание цветовой палитры для сегментов
        colors = plt.cm.RdYlGn(loyalty_counts.values / max(loyalty_counts.values))
        
        loyalty_counts.plot(kind='bar', color=colors)
        plt.title('Распределение клиентов по сегментам лояльности')
        plt.xlabel('Сегмент лояльности')
        plt.ylabel('Количество клиентов')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/loyalty_segments.png")
        plt.close()
        print("Визуализация сегментов лояльности сохранена")
    
    # Визуализация корреляций между признаками
    numeric_columns = preprocessed_df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 1:
        plt.figure(figsize=(14, 10))
        correlation_matrix = preprocessed_df[numeric_columns].corr()
        
        # Фильтрация только важных корреляций
        mask = abs(correlation_matrix) > 0.3
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=~mask, fmt=".2f", linewidths=0.5)
        plt.title('Корреляционная матрица числовых признаков (|r| > 0.3)')
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/correlation_matrix.png")
        plt.close()
        print("Корреляционная матрица сохранена")

if __name__ == "__main__":
    main() 