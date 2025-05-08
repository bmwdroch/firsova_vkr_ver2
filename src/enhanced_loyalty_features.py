#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль расширенной оценки лояльности клиентов
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_enhanced_features(df):
    """
    Создает расширенный набор признаков для оценки лояльности клиентов
    
    Аргументы:
        df (pandas.DataFrame): Агрегированный датасет на уровне клиента
        
    Возвращает:
        pandas.DataFrame: Датасет с расширенными признаками лояльности
    """
    print("Создание расширенных признаков лояльности...")
    
    # Копирование датасета
    enhanced_df = df.copy()
    
    # 1. Расчет разнообразия категорий товаров
    product_cols = [col for col in enhanced_df.columns if col.startswith('product_')]
    if product_cols:
        # Количество различных категорий, в которых клиент совершил покупки
        enhanced_df['product_diversity'] = enhanced_df[product_cols].sum(axis=1)
        
        # Индекс Джини для разнообразия покупок (0 - концентрация на одной категории, 1 - равномерное распределение)
        def gini_diversity(row):
            values = row[product_cols].values
            # Если нет покупок, возвращаем 0
            if sum(values) == 0:
                return 0
            # Нормализуем значения
            values = values / sum(values)
            # Убираем нулевые значения
            values = values[values > 0]
            # Вычисляем индекс Джини
            return 1 - sum(values**2)
        
        enhanced_df['product_diversity_gini'] = enhanced_df.apply(gini_diversity, axis=1)
        print("Добавлены признаки разнообразия категорий товаров")
    
    # 2. Анализ использования бонусной программы
    if 'bonus_usage_ratio' in enhanced_df.columns:
        # Уже есть базовый показатель использования бонусов
        
        # Дополнительно оценим активность в бонусной программе
        if 'Начислено бонусов_sum' in enhanced_df.columns and 'monetary' in enhanced_df.columns:
            # Отношение начисленных бонусов к общей сумме покупок
            enhanced_df['bonus_earning_ratio'] = enhanced_df['Начислено бонусов_sum'] / enhanced_df['monetary'].replace(0, 1)
            
        # Эффективность использования бонусной программы
        if 'Списано бонусов_sum' in enhanced_df.columns and 'Начислено бонусов_sum' in enhanced_df.columns:
            # Интегральный показатель активности в бонусной программе
            enhanced_df['bonus_activity'] = (enhanced_df['Списано бонусов_sum'] + enhanced_df['Начислено бонусов_sum']) / 2
        
        print("Добавлены расширенные признаки использования бонусной программы")
    
    # 3. Анализ активности клиента
    if 'recency' in enhanced_df.columns and 'activity_period' in enhanced_df.columns:
        # Коэффициент недавности относительно периода активности
        # (чем ближе к 0, тем более недавняя последняя покупка относительно периода активности)
        enhanced_df['recency_ratio'] = enhanced_df['recency'] / enhanced_df['activity_period'].replace(0, 1)
    
    if 'frequency' in enhanced_df.columns and 'activity_period' in enhanced_df.columns:
        # Плотность покупок в течение периода активности
        enhanced_df['purchase_density'] = enhanced_df['frequency'] / enhanced_df['activity_period'].replace(0, 1)
    
    # 4. Стабильность среднего чека
    if 'avg_purchase' in enhanced_df.columns and 'Cумма покупки_std' in enhanced_df.columns:
        # Коэффициент вариации среднего чека (относительное отклонение)
        enhanced_df['purchase_amount_cv'] = enhanced_df['Cумма покупки_std'] / enhanced_df['avg_purchase'].replace(0, 1)
        
        # Преобразуем в показатель стабильности (1 - CV, ограниченный до [0, 1])
        enhanced_df['purchase_stability'] = 1 - enhanced_df['purchase_amount_cv'].clip(0, 1)
        print("Добавлены признаки стабильности покупок")
    
    # 5. Локальная лояльность (приверженность к конкретной точке продаж)
    location_cols = [col for col in enhanced_df.columns if col.startswith('location_')]
    if location_cols:
        # Максимальная доля покупок в одной точке продаж
        enhanced_df['location_loyalty'] = enhanced_df[location_cols].max(axis=1)
        
        # Количество различных точек продаж, где совершались покупки
        enhanced_df['location_diversity'] = enhanced_df[location_cols].astype(bool).sum(axis=1)
        print("Добавлены признаки локальной лояльности")
    
    # 6. Интегральные показатели
    # RFM-баллы, если доступны
    rfm_cols = ['R', 'F', 'M']
    if all(col in enhanced_df.columns for col in rfm_cols):
        # Уже есть базовые RFM-показатели
        
        # Взвешенный RFM-скор (придаем разный вес разным компонентам)
        # Для детской одежды больший вес придадим R (частота покупок важнее из-за быстрого роста детей)
        enhanced_df['weighted_RFM'] = enhanced_df['R'] * 0.5 + enhanced_df['F'] * 0.3 + enhanced_df['M'] * 0.2
        print("Добавлен взвешенный RFM-показатель")
    
    # 7. Дополнительные признаки, если доступны соответствующие данные
    
    # Возвращаем обогащенный датасет
    print(f"Создано {len(enhanced_df.columns) - len(df.columns)} новых признаков лояльности")
    return enhanced_df

def perform_customer_clustering(df, n_clusters=5, random_state=42):
    """
    Выполняет кластеризацию клиентов на основе признаков лояльности
    
    Аргументы:
        df (pandas.DataFrame): Датасет с признаками лояльности
        n_clusters (int): Количество кластеров
        random_state (int): Seed для воспроизводимости результатов
        
    Возвращает:
        pandas.DataFrame: Датасет с добавленными метками кластеров
    """
    print(f"Выполнение кластеризации клиентов на {n_clusters} кластеров...")
    
    # Копирование датасета
    cluster_df = df.copy()
    
    # Выбор числовых признаков для кластеризации
    numeric_cols = cluster_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Исключаем ID и целевые переменные
    exclude_cols = ['Клиент', 'RFM_Score', 'RFM_Group', 'loyalty_segment', 'R', 'F', 'M']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Проверка наличия признаков
    if not feature_cols:
        print("Ошибка: Не найдены числовые признаки для кластеризации")
        return cluster_df
    
    # Подготовка данных для кластеризации
    X = cluster_df[feature_cols].copy()
    
    # Обработка пропущенных значений
    X.fillna(X.mean(), inplace=True)
    
    # Стандартизация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Определение оптимального числа кластеров с помощью метода силуэта
    if n_clusters == 'auto':
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
            print(f"Кластеры: {k}, Силуэтный коэффициент: {score:.4f}")
        
        # Выбор оптимального числа кластеров
        n_clusters = k_range[np.argmax(silhouette_scores)]
        print(f"Оптимальное число кластеров: {n_clusters}")
    
    # Применение K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Анализ кластеров
    cluster_stats = cluster_df.groupby('cluster').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'avg_purchase': 'mean',
        'Клиент': 'count'
    }).rename(columns={'Клиент': 'count'})
    
    print("Характеристики полученных кластеров:")
    print(cluster_stats)
    
    # Интерпретация кластеров
    # Маркируем кластеры на основе средних значений RFM-метрик
    
    # Кластер с низким recency (недавние покупки) и высокой frequency - лояльные клиенты
    loyalty_ranks = cluster_stats.copy()
    loyalty_ranks['recency_rank'] = loyalty_ranks['recency'].rank()
    loyalty_ranks['frequency_rank'] = loyalty_ranks['frequency'].rank(ascending=False)
    loyalty_ranks['monetary_rank'] = loyalty_ranks['monetary'].rank(ascending=False)
    loyalty_ranks['total_rank'] = loyalty_ranks['recency_rank'] + loyalty_ranks['frequency_rank'] + loyalty_ranks['monetary_rank']
    
    # Сортировка кластеров по ранжированию
    loyalty_ranks = loyalty_ranks.sort_values('total_rank')
    
    # Создаем маппинг кластеров на сегменты лояльности
    cluster_to_segment = {}
    
    loyalty_segments = [
        'Высоколояльные', 
        'Лояльные', 
        'Умеренно лояльные', 
        'Низколояльные', 
        'Отток'
    ]
    
    # Ограничиваем количество сегментов до количества кластеров
    segments_to_use = loyalty_segments[:n_clusters]
    
    for i, (cluster, _) in enumerate(loyalty_ranks.iterrows()):
        if i < len(segments_to_use):
            cluster_to_segment[cluster] = segments_to_use[i]
        else:
            cluster_to_segment[cluster] = f"Сегмент {i+1}"
    
    # Добавляем маркировку кластеров в датасет
    cluster_df['cluster_segment'] = cluster_df['cluster'].map(cluster_to_segment)
    
    print(f"Распределение кластеров по сегментам лояльности:")
    print(cluster_df['cluster_segment'].value_counts())
    
    return cluster_df

def calculate_enhanced_loyalty_score(df, pca_components=3, output_dir=None):
    """
    Рассчитывает улучшенный показатель лояльности на основе комбинации признаков
    
    Аргументы:
        df (pandas.DataFrame): Датасет с признаками лояльности и кластерами
        pca_components (int): Количество компонент для PCA
        output_dir (str): Директория для сохранения результатов
        
    Возвращает:
        pandas.DataFrame: Датасет с расчитанным показателем лояльности
    """
    print("Расчет улучшенного показателя лояльности...")
    
    # Копирование датасета
    loyalty_df = df.copy()
    
    # 1. Взвешенный RFM-скор (если RFM-метрики доступны)
    rfm_cols = ['R', 'F', 'M']
    if all(col in loyalty_df.columns for col in rfm_cols):
        if 'weighted_RFM' not in loyalty_df.columns:
            loyalty_df['weighted_RFM'] = loyalty_df['R'] * 0.5 + loyalty_df['F'] * 0.3 + loyalty_df['M'] * 0.2
    
    # 2. Интегральный показатель на основе PCA
    
    # Выбор числовых признаков для PCA
    numeric_cols = loyalty_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Исключаем ID, метки кластеров и целевые переменные
    exclude_cols = [
        'Клиент', 'cluster', 'RFM_Score', 'RFM_Group', 'loyalty_segment', 
        'weighted_RFM', 'enhanced_loyalty_score', 'cluster_segment'
    ]
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and not col.startswith('Дата')]
    
    # Проверка наличия признаков
    if not feature_cols:
        print("Ошибка: Не найдены числовые признаки для PCA")
        return loyalty_df
    
    # Подготовка данных для PCA
    X = loyalty_df[feature_cols].copy()
    
    # Обработка пропущенных значений
    X.fillna(X.mean(), inplace=True)
    
    # Стандартизация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Применение PCA для снижения размерности
    n_components = min(pca_components, len(feature_cols), len(X))
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    
    # Вывод объясненной дисперсии
    explained_variance = pca.explained_variance_ratio_
    print(f"Объясненная дисперсия по компонентам PCA: {explained_variance}")
    
    # Создание PCA-признаков
    for i in range(n_components):
        loyalty_df[f'pca_component_{i+1}'] = pca_result[:, i]
    
    # 3. Создание комбинированного показателя лояльности
    
    # Если есть взвешенный RFM-скор, используем его как базу
    if 'weighted_RFM' in loyalty_df.columns:
        # Нормализация RFM-скора к диапазону [0, 1]
        max_rfm = 12  # Максимальное значение RFM-скора (4+4+4)
        loyalty_df['rfm_normalized'] = loyalty_df['weighted_RFM'] / max_rfm
        
        # Базовая составляющая лояльности
        loyalty_base = loyalty_df['rfm_normalized']
    else:
        # Если RFM недоступен, используем первую компоненту PCA
        # Нормализация к диапазону [0, 1]
        pca1_min = loyalty_df['pca_component_1'].min()
        pca1_max = loyalty_df['pca_component_1'].max()
        loyalty_df['pca1_normalized'] = (loyalty_df['pca_component_1'] - pca1_min) / (pca1_max - pca1_min)
        
        # Базовая составляющая лояльности
        loyalty_base = loyalty_df['pca1_normalized']
    
    # Если доступны дополнительные признаки, учитываем их
    loyalty_bonus = 0
    
    # Бонус за активное использование бонусов
    if 'bonus_usage_ratio' in loyalty_df.columns:
        loyalty_df['bonus_usage_normalized'] = loyalty_df['bonus_usage_ratio'].clip(0, 1)
        loyalty_bonus += loyalty_df['bonus_usage_normalized'] * 0.15
    
    # Бонус за разнообразие категорий
    if 'product_diversity_gini' in loyalty_df.columns:
        loyalty_bonus += loyalty_df['product_diversity_gini'] * 0.1
    
    # Бонус за стабильность покупок
    if 'purchase_stability' in loyalty_df.columns:
        loyalty_bonus += loyalty_df['purchase_stability'] * 0.15
    
    # Бонус за локальную лояльность
    if 'location_loyalty' in loyalty_df.columns:
        loyalty_bonus += loyalty_df['location_loyalty'] * 0.1
    
    # Штраф за давность последней покупки
    if 'recency_ratio' in loyalty_df.columns:
        recency_penalty = loyalty_df['recency_ratio'].clip(0, 1) * 0.2
        loyalty_bonus -= recency_penalty
    
    # Расчет итогового показателя лояльности
    loyalty_df['enhanced_loyalty_score'] = (loyalty_base * 0.7 + loyalty_bonus).clip(0, 1)
    
    # 4. Создание категорий лояльности на основе улучшенного показателя
    
    # Определение границ категорий на основе процентилей
    thresholds = loyalty_df['enhanced_loyalty_score'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    
    def assign_loyalty_category(score):
        if score >= thresholds[3]:
            return 'Высоколояльные'
        elif score >= thresholds[2]:
            return 'Лояльные'
        elif score >= thresholds[1]:
            return 'Умеренно лояльные'
        elif score >= thresholds[0]:
            return 'Низколояльные'
        else:
            return 'Отток'
    
    loyalty_df['enhanced_loyalty_category'] = loyalty_df['enhanced_loyalty_score'].apply(assign_loyalty_category)
    
    # Сравнение с исходной сегментацией RFM
    if 'loyalty_segment' in loyalty_df.columns:
        match_percent = (loyalty_df['enhanced_loyalty_category'] == loyalty_df['loyalty_segment']).mean() * 100
        print(f"Соответствие с исходной RFM-сегментацией: {match_percent:.2f}%")
    
    # Сравнение с кластеризацией
    if 'cluster_segment' in loyalty_df.columns:
        match_percent = (loyalty_df['enhanced_loyalty_category'] == loyalty_df['cluster_segment']).mean() * 100
        print(f"Соответствие с сегментацией по кластерам: {match_percent:.2f}%")
    
    # Визуализация распределения улучшенного показателя лояльности
    if output_dir:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        sns.histplot(loyalty_df['enhanced_loyalty_score'], kde=True)
        plt.title('Распределение улучшенного показателя лояльности')
        
        plt.subplot(2, 2, 2)
        sns.countplot(x='enhanced_loyalty_category', data=loyalty_df, order=[
            'Высоколояльные', 'Лояльные', 'Умеренно лояльные', 'Низколояльные', 'Отток'
        ])
        plt.title('Распределение по категориям лояльности')
        plt.xticks(rotation=45)
        
        # Сравнение с исходной сегментацией RFM
        if 'loyalty_segment' in loyalty_df.columns:
            plt.subplot(2, 2, 3)
            confusion_matrix = pd.crosstab(
                loyalty_df['loyalty_segment'], 
                loyalty_df['enhanced_loyalty_category'],
                normalize='index'
            )
            sns.heatmap(confusion_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
            plt.title('Сравнение с исходной RFM-сегментацией')
            plt.tight_layout()
        
        # Сохранение визуализации
        vis_path = os.path.join(output_dir, 'enhanced_loyalty_visualization.png')
        plt.savefig(vis_path)
        print(f"Визуализация сохранена в {vis_path}")
        plt.close()
    
    print(f"Расчет улучшенного показателя лояльности завершен")
    print(f"Распределение по категориям лояльности:")
    print(loyalty_df['enhanced_loyalty_category'].value_counts())
    
    return loyalty_df

def prepare_final_loyalty_dataset(df, output_dir=None):
    """
    Подготовка итогового датасета с улучшенными признаками лояльности
    
    Аргументы:
        df (pandas.DataFrame): Датасет с расширенными признаками и улучшенным показателем лояльности
        output_dir (str): Директория для сохранения результатов
        
    Возвращает:
        pandas.DataFrame: Финальный датасет, готовый для моделирования
    """
    print("Подготовка итогового датасета с улучшенными признаками лояльности...")
    
    # Копирование датасета
    final_df = df.copy()
    
    # Создание новой целевой переменной для задачи классификации
    if 'enhanced_loyalty_category' in final_df.columns:
        # Преобразуем категории в числовые метки
        loyalty_map = {
            'Высоколояльные': 4,
            'Лояльные': 3,
            'Умеренно лояльные': 2,
            'Низколояльные': 1,
            'Отток': 0
        }
        final_df['loyalty_target'] = final_df['enhanced_loyalty_category'].map(loyalty_map)
    
    # Оценка важности признаков для определения лояльности
    if 'loyalty_target' in final_df.columns:
        # Выбираем числовые признаки для анализа
        numeric_cols = final_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Исключаем целевые переменные и метки
        exclude_cols = [
            'Клиент', 'loyalty_target', 'cluster', 'RFM_Score', 
            'R', 'F', 'M', 'enhanced_loyalty_score'
        ]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols and not col.startswith('Дата')]
        
        # Расчет корреляции с целевой переменной
        correlations = final_df[feature_cols + ['loyalty_target']].corr()['loyalty_target'].sort_values(ascending=False)
        
        print("Топ-10 признаков по корреляции с целевой переменной:")
        print(correlations.head(10))
        
        # Визуализация важности признаков
        if output_dir:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(1, 1, 1)
            top_features = correlations.drop('loyalty_target').abs().nlargest(15).index
            sns.barplot(x=correlations[top_features].values, y=top_features)
            plt.title('Важность признаков для определения лояльности')
            plt.tight_layout()
            
            # Сохранение визуализации
            vis_path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(vis_path)
            print(f"Визуализация важности признаков сохранена в {vis_path}")
            plt.close()
    
    # Сохранение итогового датасета
    if output_dir:
        # Создаем директорию, если она не существует
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Сохраняем датасет
        output_path = os.path.join(output_dir, 'enhanced_loyalty_dataset.csv')
        final_df.to_csv(output_path, index=False)
        print(f"Итоговый датасет сохранен в {output_path}")
    
    print(f"Подготовлен итоговый датасет с улучшенными признаками лояльности")
    print(f"Размер датасета: {final_df.shape[0]} клиентов, {final_df.shape[1]} признаков")
    
    return final_df

def main():
    """
    Основная функция для запуска модуля расширенной оценки лояльности
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Расширенная оценка лояльности клиентов')
    parser.add_argument('--input', type=str, required=True, help='Путь к предобработанному CSV-файлу с данными клиентов')
    parser.add_argument('--output', type=str, default='../output', help='Директория для сохранения результатов')
    parser.add_argument('--clusters', type=int, default=5, help='Количество кластеров для сегментации клиентов')
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
    
    # Создание расширенных признаков лояльности
    enhanced_df = create_enhanced_features(df)
    
    # Кластеризация клиентов
    clustered_df = perform_customer_clustering(enhanced_df, n_clusters=args.clusters)
    
    # Расчет улучшенного показателя лояльности
    loyalty_df = calculate_enhanced_loyalty_score(clustered_df, output_dir=args.output)
    
    # Подготовка итогового датасета
    final_df = prepare_final_loyalty_dataset(loyalty_df, output_dir=args.output)
    
    print("Процесс расширенной оценки лояльности клиентов завершен успешно.")

if __name__ == "__main__":
    main() 