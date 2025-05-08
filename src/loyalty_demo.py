#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Демонстрационный скрипт для модуля расширенной оценки лояльности клиентов
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse

# Импорт модулей проекта
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import preprocess_data, prepare_final_dataset
from src.enhanced_loyalty_features import (
    create_enhanced_features, 
    perform_customer_clustering,
    calculate_enhanced_loyalty_score,
    prepare_final_loyalty_dataset
)

def load_and_preprocess_data(input_file, output_dir):
    """
    Загрузка и предобработка данных
    
    Аргументы:
        input_file (str): Путь к исходному CSV-файлу
        output_dir (str): Директория для сохранения результатов
        
    Возвращает:
        pandas.DataFrame: Предобработанный датасет
    """
    print(f"Загрузка данных из {input_file}...")
    raw_df = pd.read_csv(input_file)
    
    # Предобработка данных с помощью существующего модуля
    preprocessed_df = preprocess_data(raw_df, output_dir)
    
    # Базовая подготовка данных для RFM-анализа
    base_df = prepare_final_dataset(preprocessed_df)
    
    return base_df

def visualize_comparison(base_df, enhanced_df, output_dir):
    """
    Визуализация сравнения базового и улучшенного методов определения лояльности
    
    Аргументы:
        base_df (pandas.DataFrame): Датасет с базовым RFM-анализом
        enhanced_df (pandas.DataFrame): Датасет с улучшенной оценкой лояльности
        output_dir (str): Директория для сохранения результатов
    """
    print("Визуализация сравнения базового и улучшенного методов определения лояльности...")
    
    # Создание директории для визуализаций
    vis_dir = os.path.join(output_dir, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 1. Сравнение распределения клиентов по категориям лояльности
    plt.figure(figsize=(12, 6))
    
    # Базовое RFM-распределение
    plt.subplot(1, 2, 1)
    if 'loyalty_segment' in base_df.columns:
        order = ['Высоколояльные', 'Лояльные', 'Умеренно лояльные', 'Низколояльные', 'Отток']
        counts = base_df['loyalty_segment'].value_counts().reindex(order)
        plt.bar(counts.index, counts.values)
        plt.title('Базовый RFM-метод')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Количество клиентов')
    
    # Улучшенное распределение
    plt.subplot(1, 2, 2)
    if 'enhanced_loyalty_category' in enhanced_df.columns:
        order = ['Высоколояльные', 'Лояльные', 'Умеренно лояльные', 'Низколояльные', 'Отток']
        counts = enhanced_df['enhanced_loyalty_category'].value_counts().reindex(order)
        plt.bar(counts.index, counts.values)
        plt.title('Улучшенный метод')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'loyalty_distribution_comparison.png'))
    plt.close()
    
    # 2. Матрица сравнения категорий
    if 'loyalty_segment' in base_df.columns and 'enhanced_loyalty_category' in enhanced_df.columns:
        # Объединяем таблицы для сравнения
        merged_df = base_df[['Клиент', 'loyalty_segment']].merge(
            enhanced_df[['Клиент', 'enhanced_loyalty_category']], 
            on='Клиент', 
            how='inner'
        )
        
        # Создаем матрицу сравнения
        plt.figure(figsize=(10, 8))
        
        order = ['Высоколояльные', 'Лояльные', 'Умеренно лояльные', 'Низколояльные', 'Отток']
        comparison = pd.crosstab(
            merged_df['loyalty_segment'], 
            merged_df['enhanced_loyalty_category'],
            normalize='index'
        ).reindex(index=order, columns=order)
        
        sns.heatmap(comparison, annot=True, fmt='.2f', cmap='YlGnBu')
        plt.xlabel('Улучшенный метод')
        plt.ylabel('Базовый RFM-метод')
        plt.title('Матрица переходов между категориями лояльности')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'loyalty_confusion_matrix.png'))
        plt.close()
    
    # 3. Сравнение распределения оценок лояльности
    plt.figure(figsize=(12, 6))
    
    # Базовая RFM-оценка
    plt.subplot(1, 2, 1)
    if 'RFM_Score' in base_df.columns:
        sns.histplot(base_df['RFM_Score'], kde=True, bins=10)
        plt.title('Распределение базовой RFM-оценки')
        plt.xlabel('RFM-оценка (3-12)')
    
    # Улучшенная оценка лояльности
    plt.subplot(1, 2, 2)
    if 'enhanced_loyalty_score' in enhanced_df.columns:
        sns.histplot(enhanced_df['enhanced_loyalty_score'], kde=True, bins=10)
        plt.title('Распределение улучшенной оценки лояльности')
        plt.xlabel('Оценка лояльности (0-1)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'loyalty_score_comparison.png'))
    plt.close()
    
    # 4. Диаграмма рассеяния: RFM vs улучшенная оценка
    if 'RFM_Score' in base_df.columns and 'enhanced_loyalty_score' in enhanced_df.columns:
        # Объединяем таблицы для сравнения
        merged_df = base_df[['Клиент', 'RFM_Score']].merge(
            enhanced_df[['Клиент', 'enhanced_loyalty_score']], 
            on='Клиент', 
            how='inner'
        )
        
        plt.figure(figsize=(10, 8))
        
        # Нормализуем RFM-оценку для лучшего сравнения
        merged_df['RFM_Score_normalized'] = merged_df['RFM_Score'] / 12
        
        sns.scatterplot(
            x='RFM_Score_normalized', 
            y='enhanced_loyalty_score', 
            data=merged_df,
            alpha=0.6
        )
        plt.plot([0, 1], [0, 1], 'r--')  # Линия идеального соответствия
        plt.xlabel('Нормализованная RFM-оценка')
        plt.ylabel('Улучшенная оценка лояльности')
        plt.title('Сравнение базовой и улучшенной оценок лояльности')
        
        # Добавляем коэффициент корреляции
        corr = merged_df['RFM_Score_normalized'].corr(merged_df['enhanced_loyalty_score'])
        plt.annotate(f'Корреляция: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'loyalty_score_scatter.png'))
        plt.close()
    
    # 5. Визуализация кластеров
    if 'cluster' in enhanced_df.columns:
        # Визуализируем кластеры в пространстве RFM
        plt.figure(figsize=(15, 5))
        
        # RFM-пространство
        plt.subplot(1, 3, 1)
        if all(col in enhanced_df.columns for col in ['recency', 'frequency', 'monetary']):
            sns.scatterplot(
                x='recency', 
                y='frequency', 
                hue='cluster', 
                data=enhanced_df,
                palette='viridis',
                alpha=0.6
            )
            plt.title('Кластеры в пространстве Recency-Frequency')
        
        plt.subplot(1, 3, 2)
        if all(col in enhanced_df.columns for col in ['recency', 'monetary']):
            sns.scatterplot(
                x='recency', 
                y='monetary', 
                hue='cluster', 
                data=enhanced_df,
                palette='viridis',
                alpha=0.6
            )
            plt.title('Кластеры в пространстве Recency-Monetary')
        
        plt.subplot(1, 3, 3)
        if all(col in enhanced_df.columns for col in ['frequency', 'monetary']):
            sns.scatterplot(
                x='frequency', 
                y='monetary', 
                hue='cluster', 
                data=enhanced_df,
                palette='viridis',
                alpha=0.6
            )
            plt.title('Кластеры в пространстве Frequency-Monetary')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'clusters_rfm_space.png'))
        plt.close()
    
    # 6. Важность признаков
    pca_cols = [col for col in enhanced_df.columns if col.startswith('pca_component_')]
    if pca_cols and 'enhanced_loyalty_score' in enhanced_df.columns:
        plt.figure(figsize=(10, 6))
        
        # Корреляция между PCA-компонентами и улучшенной оценкой лояльности
        corrs = []
        for col in pca_cols:
            corr = enhanced_df[col].corr(enhanced_df['enhanced_loyalty_score'])
            corrs.append((col, abs(corr)))
        
        corrs.sort(key=lambda x: x[1], reverse=True)
        
        cols = [c[0] for c in corrs]
        values = [c[1] for c in corrs]
        
        plt.bar(cols, values)
        plt.ylabel('Абсолютная корреляция')
        plt.title('Важность PCA-компонент для улучшенной оценки лояльности')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'pca_components_importance.png'))
        plt.close()
    
    print(f"Визуализации сохранены в директории {vis_dir}")

def main():
    """
    Основная функция для запуска демонстрационного скрипта
    """
    parser = argparse.ArgumentParser(description='Демонстрация расширенной оценки лояльности клиентов')
    parser.add_argument('--input', type=str, required=True, help='Путь к исходному CSV-файлу с данными клиентов')
    parser.add_argument('--output', type=str, default='../output', help='Директория для сохранения результатов')
    parser.add_argument('--clusters', type=int, default=5, help='Количество кластеров для сегментации клиентов')
    parser.add_argument('--skip_preprocessing', action='store_true', help='Пропустить этап предобработки (использовать готовый файл)')
    args = parser.parse_args()
    
    # Создание выходной директории, если она не существует
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Путь к предобработанному файлу
    preprocessed_file = os.path.join(args.output, 'preprocessed_data.csv')
    
    # Загрузка и предобработка данных
    if args.skip_preprocessing and os.path.exists(preprocessed_file):
        print(f"Загрузка предобработанных данных из {preprocessed_file}...")
        base_df = pd.read_csv(preprocessed_file)
    else:
        base_df = load_and_preprocess_data(args.input, args.output)
        # Сохраняем предобработанные данные для последующего использования
        base_df.to_csv(preprocessed_file, index=False)
        print(f"Предобработанные данные сохранены в {preprocessed_file}")
    
    # Создание расширенных признаков лояльности
    enhanced_df = create_enhanced_features(base_df)
    
    # Кластеризация клиентов
    clustered_df = perform_customer_clustering(enhanced_df, n_clusters=args.clusters)
    
    # Расчет улучшенного показателя лояльности
    loyalty_df = calculate_enhanced_loyalty_score(clustered_df, output_dir=args.output)
    
    # Подготовка итогового датасета
    final_df = prepare_final_loyalty_dataset(loyalty_df, output_dir=args.output)
    
    # Визуализация сравнения методов
    visualize_comparison(base_df, final_df, args.output)
    
    print("Демонстрация расширенной оценки лояльности клиентов завершена успешно.")
    print(f"Результаты сохранены в директории {args.output}")

if __name__ == "__main__":
    main() 