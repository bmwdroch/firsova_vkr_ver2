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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap library not found. SHAP-based feature importance will not be available.")
    print("Install it using: pip install shap")

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn library not found. Sampling methods for class balancing will not be available.")
    print("Install it using: pip install imbalanced-learn")

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
    
    try:
        # 1. Расчет разнообразия категорий товаров
        # Выбираем только бинарные признаки продуктов, исключая текстовые столбцы
        product_cols = [col for col in enhanced_df.columns 
                       if col.startswith('product_') 
                       and col != 'product_category'
                       and pd.api.types.is_bool_dtype(enhanced_df[col])]
        
        if product_cols:
            # Количество различных категорий, в которых клиент совершил покупки
            enhanced_df['product_diversity'] = enhanced_df[product_cols].astype(float).sum(axis=1)
            
            # Индекс Джини для разнообразия покупок
            def gini_diversity(row):
                values = row[product_cols].astype(float).values
                total = np.sum(values)
                if total == 0:
                    return 0
                values = values / total
                values = values[values > 0]
                return 1 - np.sum(values ** 2)
            
            enhanced_df['product_diversity_gini'] = enhanced_df.apply(gini_diversity, axis=1)
            print(f"Добавлены признаки разнообразия категорий товаров (обработано {len(product_cols)} категорий)")
        
        # 2. Анализ использования бонусной программы
        if 'bonus_usage_ratio' in enhanced_df.columns:
            if all(col in enhanced_df.columns for col in ['Начислено бонусов_sum', 'monetary']):
                enhanced_df['bonus_earning_ratio'] = enhanced_df['Начислено бонусов_sum'] / enhanced_df['monetary'].replace(0, 1)
            
            if all(col in enhanced_df.columns for col in ['Списано бонусов_sum', 'Начислено бонусов_sum']):
                enhanced_df['bonus_activity'] = (enhanced_df['Списано бонусов_sum'] + enhanced_df['Начислено бонусов_sum']) / 2
            
            print("Добавлены расширенные признаки использования бонусной программы")
        
        # 3. Анализ активности клиента
        if all(col in enhanced_df.columns for col in ['recency', 'activity_period']):
            enhanced_df['recency_ratio'] = enhanced_df['recency'] / enhanced_df['activity_period'].replace(0, 1)
        
        if all(col in enhanced_df.columns for col in ['frequency', 'activity_period']):
            enhanced_df['purchase_density'] = enhanced_df['frequency'] / enhanced_df['activity_period'].replace(0, 1)
        
        # 4. Стабильность среднего чека
        if all(col in enhanced_df.columns for col in ['avg_purchase', 'Cумма покупки_std']):
            enhanced_df['purchase_amount_cv'] = enhanced_df['Cумма покупки_std'] / enhanced_df['avg_purchase'].replace(0, 1)
            enhanced_df['purchase_stability'] = 1 - enhanced_df['purchase_amount_cv'].clip(0, 1)
            print("Добавлены признаки стабильности покупок")
        
        # 5. Локальная лояльность
        location_cols = [col for col in enhanced_df.columns 
                        if col.startswith('location_') 
                        and pd.api.types.is_bool_dtype(enhanced_df[col])]
        
        if location_cols:
            location_data = enhanced_df[location_cols].astype(float)
            enhanced_df['location_loyalty'] = location_data.max(axis=1)
            enhanced_df['location_diversity'] = location_data.astype(bool).sum(axis=1)
            print("Добавлены признаки локальной лояльности")
        
        # 6. Интегральные показатели
        rfm_cols = ['R', 'F', 'M']
        if all(col in enhanced_df.columns for col in rfm_cols):
            enhanced_df['weighted_RFM'] = enhanced_df['R'] * 0.5 + enhanced_df['F'] * 0.3 + enhanced_df['M'] * 0.2
            print("Добавлен взвешенный RFM-показатель")
        
        print(f"Создано {len(enhanced_df.columns) - len(df.columns)} новых признаков лояльности")
        return enhanced_df
        
    except Exception as e:
        print(f"Ошибка при создании расширенных признаков: {str(e)}")
        import traceback
        traceback.print_exc()
        return df

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

def categorize_loyalty_score(df, method='quantile', n_categories=5, custom_boundaries=None, score_column='enhanced_loyalty_score'):
    """
    Преобразует непрерывный показатель лояльности в дискретные категории
    
    Аргументы:
        df (pandas.DataFrame): Датасет с показателем лояльности
        method (str): Метод категоризации ('quantile', 'kmeans', 'equal_width', 'manual')
        n_categories (int): Количество категорий
        custom_boundaries (list): Список пользовательских границ для метода 'manual'
        score_column (str): Имя столбца с показателем лояльности
        
    Возвращает:
        pandas.DataFrame: Датасет с добавленной категоризацией лояльности
    """
    print(f"Формирование дискретных категорий лояльности методом {method}...")
    
    # Копирование датасета
    result_df = df.copy()
    
    # Проверка наличия столбца с показателем лояльности
    if score_column not in result_df.columns:
        print(f"Ошибка: столбец {score_column} не найден в датасете")
        return result_df
    
    # Определение названий категорий лояльности
    if n_categories == 5:
        loyalty_categories = ['Отток', 'Низколояльные', 'Умеренно лояльные', 'Лояльные', 'Высоколояльные']
    else:
        loyalty_categories = [f'Категория {i+1}' for i in range(n_categories)]
    
    # Категоризация показателя лояльности различными методами
    category_column = f"{score_column}_category"
    
    if method == 'quantile':
        # Метод квантилей - равное количество клиентов в каждой категории
        quantiles = [i/n_categories for i in range(1, n_categories)]
        boundaries = result_df[score_column].quantile(quantiles).tolist()
        boundaries = [0] + boundaries + [1]
        
    elif method == 'kmeans':
        # Метод кластеризации K-means - естественные кластеры в данных
        kmeans = KMeans(n_clusters=n_categories, random_state=42)
        # Преобразуем в двумерный массив для K-means
        X = result_df[score_column].values.reshape(-1, 1)
        kmeans.fit(X)
        
        # Получаем центры кластеров и сортируем их
        centers = kmeans.cluster_centers_.flatten()
        sorted_centers = np.sort(centers)
        
        # Определяем границы как средние точки между центрами кластеров
        boundaries = [0]
        for i in range(len(sorted_centers)-1):
            boundary = (sorted_centers[i] + sorted_centers[i+1]) / 2
            boundaries.append(boundary)
        boundaries.append(1)
        
    elif method == 'equal_width':
        # Метод равной ширины - равные интервалы показателя лояльности
        step = 1.0 / n_categories
        boundaries = [i * step for i in range(n_categories+1)]
        
    elif method == 'manual' and custom_boundaries is not None:
        # Метод ручного задания границ
        boundaries = custom_boundaries
        if len(boundaries) != n_categories + 1:
            print(f"Ошибка: количество границ ({len(boundaries)}) не соответствует количеству категорий + 1 ({n_categories+1})")
            return result_df
    else:
        print(f"Ошибка: неподдерживаемый метод категоризации {method}")
        return result_df
    
    # Создание функции для определения категории по значению показателя
    def assign_category(score, bounds, cats):
        for i in range(len(bounds)-1):
            if bounds[i] <= score < bounds[i+1]:
                return cats[i]
        return cats[-1]  # Для максимального значения
    
    # Применение функции категоризации
    result_df[category_column] = result_df[score_column].apply(
        lambda x: assign_category(x, boundaries, loyalty_categories)
    )
    
    # Вывод информации о распределении категорий
    print("Распределение клиентов по категориям лояльности:")
    print(result_df[category_column].value_counts().sort_index())
    print(f"Границы категорий: {boundaries}")
    
    return result_df, boundaries

def get_category_profiles(df, category_column='enhanced_loyalty_score_category'):
    """
    Формирует профили категорий лояльности на основе средних значений ключевых метрик
    
    Аргументы:
        df (pandas.DataFrame): Датасет с категориями лояльности
        category_column (str): Имя столбца с категориями лояльности
        
    Возвращает:
        pandas.DataFrame: Датафрейм с профилями категорий
    """
    print("Формирование профилей категорий лояльности...")
    
    # Проверка наличия столбца с категориями лояльности
    if category_column not in df.columns:
        print(f"Ошибка: столбец {category_column} не найден в датасете")
        return None
    
    # Определение ключевых метрик для профилей
    key_metrics = []
    
    # RFM-метрики
    rfm_metrics = ['recency', 'frequency', 'monetary', 'avg_purchase']
    key_metrics.extend([m for m in rfm_metrics if m in df.columns])
    
    # Метрики бонусной программы
    bonus_metrics = ['bonus_usage_ratio', 'bonus_earning_ratio', 'bonus_activity']
    key_metrics.extend([m for m in bonus_metrics if m in df.columns])
    
    # Метрики активности
    activity_metrics = ['recency_ratio', 'purchase_density', 'activity_period']
    key_metrics.extend([m for m in activity_metrics if m in df.columns])
    
    # Метрики разнообразия и стабильности
    diversity_metrics = ['product_diversity', 'product_diversity_gini', 'purchase_stability', 'location_loyalty']
    key_metrics.extend([m for m in diversity_metrics if m in df.columns])
    
    # Если метрик не найдено, используем все числовые столбцы
    if not key_metrics:
        key_metrics = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        key_metrics = [col for col in key_metrics if col != category_column and not col.endswith('_category')]
    
    # Группировка по категориям и расчет средних значений метрик
    profiles = df.groupby(category_column)[key_metrics].mean()
    
    # Добавление размера категорий
    category_sizes = df[category_column].value_counts()
    profiles['размер_категории'] = category_sizes
    profiles['доля_категории'] = category_sizes / len(df)
    
    print("Сформированы профили категорий лояльности")
    
    return profiles

def evaluate_category_distribution(df, category_column='enhanced_loyalty_score_category'):
    """
    Оценивает распределение клиентов по категориям лояльности
    и выявляет потенциальные проблемы с балансом
    
    Аргументы:
        df (pandas.DataFrame): Датасет с категориями лояльности
        category_column (str): Имя столбца с категориями лояльности
        
    Возвращает:
        dict: Словарь с оценками распределения категорий
    """
    print("Оценка распределения клиентов по категориям лояльности...")
    
    # Проверка наличия столбца с категориями лояльности
    if category_column not in df.columns:
        print(f"Ошибка: столбец {category_column} не найден в датасете")
        return None
    
    # Расчет распределения клиентов по категориям
    category_counts = df[category_column].value_counts().sort_index()
    category_percents = category_counts / len(df) * 100
    
    # Оценка сбалансированности классов
    total_clients = len(df)
    num_categories = len(category_counts)
    expected_per_category = total_clients / num_categories
    max_imbalance = category_counts.max() / category_counts.min()
    
    # Расчет энтропии распределения (мера сбалансированности)
    probabilities = category_counts / total_clients
    entropy = -np.sum(probabilities * np.log2(probabilities))
    max_entropy = np.log2(num_categories)  # Максимальная энтропия при равномерном распределении
    normalized_entropy = entropy / max_entropy
    
    # Подготовка результата
    evaluation = {
        'total_clients': total_clients,
        'num_categories': num_categories,
        'category_counts': category_counts.to_dict(),
        'category_percents': category_percents.to_dict(),
        'max_imbalance': max_imbalance,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'is_balanced': normalized_entropy > 0.8  # Пороговое значение сбалансированности
    }
    
    # Вывод основной информации
    print(f"Общее количество клиентов: {total_clients}")
    print(f"Количество категорий: {num_categories}")
    print("Распределение клиентов по категориям:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} ({category_percents[category]:.2f}%)")
    
    print(f"Максимальный дисбаланс: {max_imbalance:.2f}:1")
    print(f"Нормализованная энтропия распределения: {normalized_entropy:.4f} (из 1.0)")
    
    if evaluation['is_balanced']:
        print("Оценка: Распределение категорий достаточно сбалансировано")
    else:
        print("Оценка: Распределение категорий имеет значительный дисбаланс")
        
        # Предложения по улучшению баланса
        if max_imbalance > 3:
            print("Рекомендации:")
            print("  - Рассмотрите другой метод формирования категорий (например, квантили)")
            print("  - Объедините малочисленные категории")
            print("  - Разделите многочисленные категории на подкатегории")
    
    return evaluation

def optimize_category_boundaries(df, initial_boundaries, target_distribution=None, score_column='enhanced_loyalty_score'):
    """
    Оптимизирует границы категорий лояльности для достижения целевого распределения
    
    Аргументы:
        df (pandas.DataFrame): Датасет с показателем лояльности
        initial_boundaries (list): Исходные границы категорий
        target_distribution (list): Целевое распределение клиентов по категориям (в %)
        score_column (str): Имя столбца с показателем лояльности
        
    Возвращает:
        list: Оптимизированные границы категорий
    """
    print("Оптимизация границ категорий лояльности...")
    
    # Проверка наличия столбца с показателем лояльности
    if score_column not in df.columns:
        print(f"Ошибка: столбец {score_column} не найден в датасете")
        return initial_boundaries
    
    # Если целевое распределение не задано, используем равномерное
    if target_distribution is None:
        num_categories = len(initial_boundaries) - 1
        target_distribution = [100 / num_categories] * num_categories
    
    # Проверка корректности целевого распределения
    if sum(target_distribution) != 100 or len(target_distribution) != len(initial_boundaries) - 1:
        print("Ошибка: целевое распределение должно в сумме давать 100% и соответствовать количеству категорий")
        return initial_boundaries
    
    # Копирование датасета
    scores = df[score_column].values
    scores.sort()
    
    # Расчет границ для целевого распределения
    optimized_boundaries = [0]
    cumulative = 0
    for target_percent in target_distribution[:-1]:  # Последняя граница всегда равна 1
        cumulative += target_percent
        idx = int(len(scores) * cumulative / 100)
        if idx < len(scores):
            optimized_boundaries.append(scores[idx])
        else:
            optimized_boundaries.append(1)
    optimized_boundaries.append(1)
    
    # Округление границ до 2 знаков после запятой для удобства интерпретации
    optimized_boundaries = [round(b, 2) for b in optimized_boundaries]
    
    print(f"Исходные границы: {initial_boundaries}")
    print(f"Оптимизированные границы: {optimized_boundaries}")
    
    return optimized_boundaries

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
    
    # Если есть хотя бы 3 числовых признака, применяем PCA
    if len(feature_cols) >= 3:
        # Масштабирование данных
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(loyalty_df[feature_cols])
        
        # Применение PCA
        pca = PCA(n_components=min(pca_components, len(feature_cols)))
        pca_result = pca.fit_transform(scaled_features)
        
        # Добавление PCA компонент в датасет
        for i in range(pca_result.shape[1]):
            loyalty_df[f'pca_component_{i+1}'] = pca_result[:, i]
        
        # Вывод информации о PCA
        explained_variance = pca.explained_variance_ratio_
        print(f"PCA выполнен. Объясненная дисперсия по компонентам:")
        for i, var in enumerate(explained_variance):
            print(f"Компонента {i+1}: {var:.4f} ({var*100:.2f}%)")
        print(f"Суммарная объясненная дисперсия: {sum(explained_variance)*100:.2f}%")
        
        # Визуализация PCA компонент
        if output_dir is not None:
            vis_dir = os.path.join(output_dir, 'visualizations')
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            
            # Scatter plot для первых двух компонент PCA
            plt.figure(figsize=(10, 8))
            
            # Если есть кластеры, используем их для цвета
            if 'cluster' in loyalty_df.columns:
                scatter = plt.scatter(
                    loyalty_df['pca_component_1'], 
                    loyalty_df['pca_component_2'],
                    c=loyalty_df['cluster'], 
                    cmap='viridis', 
                    alpha=0.6,
                    s=50
                )
                plt.colorbar(scatter, label='Кластер')
            else:
                plt.scatter(
                    loyalty_df['pca_component_1'], 
                    loyalty_df['pca_component_2'],
                    alpha=0.6,
                    s=50
                )
            
            plt.title('PCA: первые две главные компоненты')
            plt.xlabel(f'PCA 1 ({explained_variance[0]*100:.2f}%)')
            plt.ylabel(f'PCA 2 ({explained_variance[1]*100:.2f}%)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'pca_components.png'))
            plt.close()
    
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
    
    # Используем функцию categorize_loyalty_score для создания категорий
    loyalty_df, boundaries = categorize_loyalty_score(
        loyalty_df, 
        method='quantile',  # По умолчанию используем метод квантилей
        n_categories=5,
        score_column='enhanced_loyalty_score'
    )
    
    # Оценка распределения категорий
    category_evaluation = evaluate_category_distribution(loyalty_df, 'enhanced_loyalty_score_category')
    
    # Если распределение сильно несбалансировано, предложить оптимизацию
    if category_evaluation and not category_evaluation['is_balanced']:
        print("Обнаружен дисбаланс в распределении категорий.")
        print("Рекомендуется оптимизировать границы категорий.")
    
    # Сравнение с исходной сегментацией RFM
    if 'loyalty_segment' in loyalty_df.columns:
        match_percent = (loyalty_df['enhanced_loyalty_score_category'] == loyalty_df['loyalty_segment']).mean() * 100
        print(f"Соответствие с исходной RFM-сегментацией: {match_percent:.2f}%")
    
    # Создание профилей категорий
    category_profiles = get_category_profiles(loyalty_df, 'enhanced_loyalty_score_category')
    print("\nПрофили категорий лояльности:")
    print(category_profiles)
    
    # Визуализация распределения категорий
    if output_dir is not None:
        vis_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        
        # Распределение категорий
        plt.figure(figsize=(10, 6))
        category_counts = loyalty_df['enhanced_loyalty_score_category'].value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0, 1, len(category_counts)))
        category_counts.plot(kind='bar', color=colors)
        plt.title('Распределение клиентов по категориям лояльности')
        plt.xlabel('Категория лояльности')
        plt.ylabel('Количество клиентов')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'loyalty_categories_distribution.png'))
        plt.close()
        
        # Распределение показателя лояльности с отмеченными границами категорий
        plt.figure(figsize=(12, 6))
        sns.histplot(loyalty_df['enhanced_loyalty_score'], bins=50, kde=True)
        plt.title('Распределение показателя лояльности')
        plt.xlabel('Показатель лояльности')
        plt.ylabel('Количество клиентов')
        
        # Отмечаем границы категорий
        for i, bound in enumerate(boundaries):
            if i > 0 and i < len(boundaries) - 1:  # Пропускаем 0 и 1
                plt.axvline(x=bound, color='r', linestyle='--', alpha=0.7)
                plt.text(bound, plt.ylim()[1]*0.9, f'{bound:.2f}', 
                         rotation=90, verticalalignment='top')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'loyalty_score_distribution.png'))
        plt.close()
    
    print(f"Расчет улучшенного показателя лояльности завершен")
    print(f"Распределение по категориям лояльности:")
    print(loyalty_df['enhanced_loyalty_score_category'].value_counts())
    
    return loyalty_df

def balance_loyalty_classes(df, target_column, method='random_over', sampling_strategy='auto', 
                           random_state=42, features=None, return_indices=False):
    """
    Балансирует классы в целевом признаке с использованием различных методов сэмплирования
    
    Аргументы:
        df (pandas.DataFrame): Исходный датасет
        target_column (str): Имя столбца с целевым признаком (классами)
        method (str): Метод сэмплирования ('random_over', 'random_under', 'smote', 'combine')
        sampling_strategy: Стратегия сэмплирования ('auto', dict или float)
        random_state (int): Случайное зерно для воспроизводимости
        features (list): Список признаков для использования при создании синтетических примеров
        return_indices (bool): Возвращать индексы выбранных/созданных примеров
        
    Возвращает:
        pandas.DataFrame: Датасет с балансированными классами
    """
    print(f"Балансировка классов методом {method}...")
    
    # Проверка наличия библиотеки imblearn
    if not IMBLEARN_AVAILABLE:
        print("Ошибка: библиотека imbalanced-learn не установлена. Установите её с помощью: pip install imbalanced-learn")
        return df
    
    # Проверка наличия целевой переменной
    if target_column not in df.columns:
        print(f"Ошибка: столбец {target_column} не найден в датасете")
        return df
    
    # Анализ исходного распределения классов
    original_counts = df[target_column].value_counts().sort_index()
    print("Исходное распределение классов:")
    for cls, count in original_counts.items():
        print(f"  Класс {cls}: {count} ({count/len(df)*100:.2f}%)")
    
    # Выбор признаков для сэмплирования
    if features is None:
        # Если признаки не указаны, используем все числовые признаки кроме целевой переменной
        features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        features = [col for col in features if col != target_column]
    
    # Проверка наличия выбранных признаков
    if not features:
        print("Ошибка: не найдены признаки для балансировки")
        return df
    
    # Подготовка данных для балансировки
    X = df[features].copy()
    y = df[target_column].copy()
    
    # Обработка пропущенных значений
    X.fillna(X.mean(), inplace=True)
    
    # Выбор и применение метода балансировки
    if method == 'random_over':
        # Случайный oversampling
        print("Применение Random Oversampling...")
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
    elif method == 'random_under':
        # Случайный undersampling
        print("Применение Random Undersampling...")
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
    elif method == 'smote':
        # SMOTE
        print("Применение SMOTE...")
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
    elif method == 'combine':
        # Комбинация SMOTE и Tomek links
        print("Применение комбинированного метода (SMOTE + Tomek)...")
        sampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
    else:
        print(f"Ошибка: неизвестный метод балансировки {method}")
        return df
    
    # Создание нового датафрейма с балансированными данными
    resampled_df = pd.DataFrame(X_resampled, columns=features)
    resampled_df[target_column] = y_resampled
    
    # Если необходимо сохранить индексы
    if return_indices and hasattr(sampler, 'sample_indices_'):
        return resampled_df, sampler.sample_indices_
    
    # Анализ нового распределения классов
    balanced_counts = resampled_df[target_column].value_counts().sort_index()
    print("Распределение классов после балансировки:")
    for cls, count in balanced_counts.items():
        print(f"  Класс {cls}: {count} ({count/len(resampled_df)*100:.2f}%)")
    
    # Расчет улучшения баланса
    max_imbalance_before = original_counts.max() / original_counts.min()
    max_imbalance_after = balanced_counts.max() / balanced_counts.min()
    print(f"Коэффициент дисбаланса уменьшился с {max_imbalance_before:.2f}:1 до {max_imbalance_after:.2f}:1")
    
    return resampled_df

def evaluate_feature_importance(X_features, y_target, methods=None, n_estimators=100, 
                              n_repeats=10, n_top_features=20, random_state=42, output_dir=None):
    """
    Оценивает информативность признаков с использованием различных методов
    
    Аргументы:
        X_features (pandas.DataFrame): Датафрейм с признаками (только признаки)
        y_target (pandas.Series): Серия с целевой переменной
        methods (list): Список методов для оценки важности признаков
                      ('correlation', 'model_based', 'permutation', 'shap')
        n_estimators (int): Количество деревьев для Random Forest
        n_repeats (int): Количество повторений для permutation importance
        n_top_features (int): Количество наиболее важных признаков для вывода
        random_state (int): Случайное зерно для воспроизводимости
        output_dir (str): Директория для сохранения результатов
        
    Возвращает:
        pandas.DataFrame: Датафрейм с оценками важности признаков
    """
    print("Оценка информативности признаков...")
    
    # Проверка наличия целевой переменной - теперь y_target это Series, проверка на None или пустоту
    if y_target is None or y_target.empty:
        print(f"Ошибка: целевая переменная (y_target) не предоставлена или пуста")
        return None
    
    # Определение методов по умолчанию
    if methods is None:
        methods = ['correlation', 'model_based', 'permutation']
        if SHAP_AVAILABLE:
            methods.append('shap')
    
    # Подготовка результирующего датафрейма
    all_importances = pd.DataFrame()
    
    # Подготовка данных
    # X теперь это X_features, y это y_target
    X = X_features.copy() 
    y = y_target.copy()
    
    # Исключаем целевую переменную и служебные столбцы - УЖЕ СДЕЛАНО ПРИ СОЗДАНИИ X_features
    # exclude_cols = [target_column, 'Клиент', 'original_index', 'is_synthetic']
    # exclude_cols.extend([col for col in X.columns if col.endswith('_target')])
    
    features = X.columns.tolist() # Получаем имена признаков из X_features
    
    # Проверка наличия признаков
    if not features:
        print("Ошибка: не найдены числовые признаки для анализа")
        return None
    
    # 1. Корреляционный анализ
    if 'correlation' in methods:
        print("Вычисление корреляций Пирсона...")
        # Вычисление корреляций с целевой переменной
        corr_importances = pd.DataFrame()
        corr_importances['feature'] = features
        
        # Абсолютные значения корреляции Пирсона
        # if df[target_column].dtype in ['int64', 'float64']: # Теперь y это Series
        if y.dtype in ['int64', 'float64']:
            corrs = []
            for feature_name in features: # Используем feature_name для итерации, чтобы не конфликтовать с features list
                corr = abs(X[feature_name].corr(y)) # X[feature_name] вместо df[feature]
                corrs.append(corr)
            corr_importances['pearson_corr'] = corrs
        
        # Взаимная информация (для категориальных целевых переменных)
        print("Вычисление взаимной информации...")
        try:
            mi_scores = mutual_info_classif(X, y, random_state=random_state)
            corr_importances['mutual_info'] = mi_scores
            
            # Нормализация оценок взаимной информации
            corr_importances['mutual_info_norm'] = corr_importances['mutual_info'] / corr_importances['mutual_info'].max()
            
            # Объединенная метрика для сортировки
            if 'pearson_corr' in corr_importances.columns:
                corr_importances['combined_score'] = (corr_importances['pearson_corr'] + corr_importances['mutual_info_norm']) / 2
            else:
                corr_importances['combined_score'] = corr_importances['mutual_info_norm']
            
            # Сортировка по комбинированной оценке
            corr_importances = corr_importances.sort_values('combined_score', ascending=False).reset_index(drop=True)
            
            all_importances = pd.concat([all_importances, corr_importances], axis=1)
        except Exception as e:
            print(f"Ошибка при вычислении взаимной информации: {e}")
    
    # 2. Важность признаков на основе модели (Random Forest)
    if 'model_based' in methods:
        print("Вычисление важности признаков на основе Random Forest...")
        try:
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=random_state
            )
            
            # Обучение Random Forest
            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            rf.fit(X_train, y_train)
            
            # Получение важности признаков
            rf_importances = pd.DataFrame()
            rf_importances['feature'] = features
            rf_importances['rf_importance'] = rf.feature_importances_
            
            # Нормализация оценок важности
            rf_importances['rf_importance_norm'] = rf_importances['rf_importance'] / rf_importances['rf_importance'].max()
            
            # Сортировка по важности
            rf_importances = rf_importances.sort_values('rf_importance', ascending=False).reset_index(drop=True)
            
            # Добавление в общую таблицу
            if all_importances.empty:
                all_importances = rf_importances
            else:
                # Соединение с существующими результатами
                all_importances = all_importances.merge(
                    rf_importances[['feature', 'rf_importance', 'rf_importance_norm']], 
                    on='feature', 
                    how='outer'
                )
        except Exception as e:
            print(f"Ошибка при вычислении важности признаков RF: {e}")
    
    # 3. Permutation Importance
    if 'permutation' in methods:
        print("Вычисление Permutation Importance...")
        try:
            # Используем тот же Random Forest
            if 'model_based' not in methods:
                # Если RF еще не обучен, обучаем его
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=random_state
                )
                rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                rf.fit(X_train, y_train)
            
            # Вычисляем permutation importance
            perm_importance = permutation_importance(
                rf, X_test, y_test, n_repeats=n_repeats, random_state=random_state
            )
            
            # Создаем датафрейм с результатами
            perm_importances = pd.DataFrame()
            perm_importances['feature'] = features
            perm_importances['perm_importance'] = perm_importance.importances_mean
            
            # Нормализация оценок важности
            perm_importances['perm_importance_norm'] = perm_importances['perm_importance'] / perm_importances['perm_importance'].max()
            
            # Сортировка по важности
            perm_importances = perm_importances.sort_values('perm_importance', ascending=False).reset_index(drop=True)
            
            # Добавление в общую таблицу
            if all_importances.empty:
                all_importances = perm_importances
            else:
                # Соединение с существующими результатами
                all_importances = all_importances.merge(
                    perm_importances[['feature', 'perm_importance', 'perm_importance_norm']], 
                    on='feature', 
                    how='outer'
                )
        except Exception as e:
            print(f"Ошибка при вычислении Permutation Importance: {e}")
    
    # 4. SHAP Values
    if 'shap' in methods and SHAP_AVAILABLE:
        print("Вычисление SHAP Values...")
        try:
            # Используем тот же Random Forest
            if 'model_based' not in methods and 'permutation' not in methods:
                # Если RF еще не обучен, обучаем его
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=random_state
                )
                rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                rf.fit(X_train, y_train)
            
            # Создаем SHAP explainer
            explainer = shap.TreeExplainer(rf)
            
            # Вычисляем SHAP values для тестовой выборки
            shap_values = explainer.shap_values(X_test)
            
            # Вычисляем средние абсолютные SHAP values для каждого признака
            if isinstance(shap_values, list):  # Для мультиклассовой задачи
                # Для многоклассовой задачи берем средние значения по всем классам
                mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            print(f"DEBUG SHAP: shape of mean_abs_shap: {mean_abs_shap.shape if isinstance(mean_abs_shap, np.ndarray) else type(mean_abs_shap)}")
            print(f"DEBUG SHAP: length of features list: {len(features)}")

            # Создаем датафрейм с результатами
            shap_importances = pd.DataFrame()
            shap_importances['feature'] = features
            shap_importances['shap_importance'] = mean_abs_shap
            
            # Нормализация оценок важности
            max_shap_val = shap_importances['shap_importance'].max()
            if max_shap_val > 0:
                shap_importances['shap_importance_norm'] = shap_importances['shap_importance'] / max_shap_val
            else:
                shap_importances['shap_importance_norm'] = 0.0 # если все значения 0 или отрицательные (хотя abs)
            
            # Сортировка по важности
            shap_importances = shap_importances.sort_values('shap_importance', ascending=False).reset_index(drop=True)
            
            # Добавление в общую таблицу
            if all_importances.empty:
                all_importances = shap_importances
            else:
                # Соединение с существующими результатами
                all_importances = all_importances.merge(
                    shap_importances[['feature', 'shap_importance', 'shap_importance_norm']], 
                    on='feature', 
                    how='outer'
                )
                
            # Визуализация SHAP values
            if output_dir is not None:
                vis_dir = os.path.join(output_dir, 'visualizations')
                if not os.path.exists(vis_dir):
                    os.makedirs(vis_dir)
                
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'shap_summary.png'))
                plt.close()
                
                # Для наиболее важных признаков создаем SHAP dependence plot
                top_features = shap_importances['feature'].head(3).tolist()
                for feature in top_features:
                    plt.figure(figsize=(8, 6))
                    
                    # Для многоклассовой задачи показываем зависимости для первого класса
                    if isinstance(shap_values, list):
                        shap.dependence_plot(feature, shap_values[0], X_test, show=False)
                    else:
                        shap.dependence_plot(feature, shap_values, X_test, show=False)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f'shap_dependence_{feature}.png'))
                    plt.close()
                    
        except Exception as e:
            print(f"Ошибка при вычислении SHAP Values: {e}")
    
    # Создание интегральной оценки важности признаков
    if not all_importances.empty:
        # Определение столбцов с нормализованными оценками
        norm_cols = [col for col in all_importances.columns if col.endswith('_norm')]
        
        if norm_cols:
            # Создание интегральной оценки как среднего нормализованных оценок
            all_importances['importance_score'] = all_importances[norm_cols].mean(axis=1)
            
            # Сортировка по интегральной оценке
            all_importances = all_importances.sort_values('importance_score', ascending=False).reset_index(drop=True)
    
    # Визуализация результатов
    if output_dir is not None and not all_importances.empty:
        vis_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        
        # Ограничиваем число признаков для визуализации
        top_features_df = all_importances.head(min(n_top_features, len(all_importances)))
        
        # Визуализация интегральной оценки важности признаков
        if 'importance_score' in all_importances.columns:
            plt.figure(figsize=(12, min(10, len(top_features_df)*0.4)))
            sns.barplot(x='importance_score', y='feature', data=top_features_df, palette='viridis')
            plt.title(f'Топ-{len(top_features_df)} наиболее важных признаков')
            plt.xlabel('Интегральная оценка важности')
            plt.ylabel('Признак')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'feature_importance_integrated.png'))
            plt.close()
        
        # Визуализация важности признаков по различным методам
        for method in methods:
            if method == 'correlation' and 'combined_score' in all_importances.columns:
                plt.figure(figsize=(12, min(10, len(top_features_df)*0.4)))
                sns.barplot(x='combined_score', y='feature', data=top_features_df, palette='Blues_r')
                plt.title(f'Топ-{len(top_features_df)} признаков по корреляциям')
                plt.xlabel('Комбинированная оценка корреляции')
                plt.ylabel('Признак')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'feature_importance_correlation.png'))
                plt.close()
                
            if method == 'model_based' and 'rf_importance' in all_importances.columns:
                plt.figure(figsize=(12, min(10, len(top_features_df)*0.4)))
                sns.barplot(x='rf_importance', y='feature', data=top_features_df.sort_values('rf_importance', ascending=False).head(n_top_features), palette='Greens_r')
                plt.title(f'Топ-{len(top_features_df)} признаков по важности Random Forest')
                plt.xlabel('Важность Random Forest')
                plt.ylabel('Признак')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'feature_importance_rf.png'))
                plt.close()
                
            if method == 'permutation' and 'perm_importance' in all_importances.columns:
                plt.figure(figsize=(12, min(10, len(top_features_df)*0.4)))
                sns.barplot(x='perm_importance', y='feature', data=top_features_df.sort_values('perm_importance', ascending=False).head(n_top_features), palette='Reds_r')
                plt.title(f'Топ-{len(top_features_df)} признаков по Permutation Importance')
                plt.xlabel('Permutation Importance')
                plt.ylabel('Признак')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'feature_importance_permutation.png'))
                plt.close()
    
    # Сохранение результатов
    if output_dir is not None and not all_importances.empty:
        results_path = os.path.join(output_dir, 'feature_importance.csv')
        all_importances.to_csv(results_path, index=False)
        print(f"Результаты оценки важности признаков сохранены в {results_path}")
    
    print(f"Оценка информативности признаков завершена. Проанализировано {len(features)} признаков.")
    
    if not all_importances.empty:
        print("Топ-10 наиболее важных признаков:")
        for i, row in all_importances.head(10).iterrows():
            print(f"{i+1}. {row['feature']}: {row.get('importance_score', 0):.4f}")
    
    return all_importances

def select_important_features(df_with_features, importance_df, min_importance=0.05, max_features=30, method='score'):
    """
    Выбирает наиболее информативные признаки на основе результатов оценки информативности
    
    Аргументы:
        df_with_features (pandas.DataFrame): Исходный датасет (или датафрейм, содержащий как минимум колонки из importance_df['feature'])
        importance_df (pandas.DataFrame): Датафрейм с оценками важности признаков
        min_importance (float): Минимальный порог важности для выбора признака
        max_features (int): Максимальное количество признаков для выбора
        method (str): Метод выбора ('score', 'correlation', 'model', 'permutation', 'shap')
        
    Возвращает:
        list: Список отобранных наиболее важных признаков
    """
    print(f"Выбор наиболее информативных признаков (метод: {method}, порог: {min_importance}, макс: {max_features})...")
    
    if importance_df is None or len(importance_df) == 0:
        print("Ошибка: Не предоставлены данные об информативности признаков")
        return None
    
    # Определение столбца для сортировки
    if method == 'score' and 'importance_score' in importance_df.columns:
        score_column = 'importance_score'
    elif method == 'correlation' and 'combined_score' in importance_df.columns:
        score_column = 'combined_score'
    elif method == 'model' and 'rf_importance' in importance_df.columns:
        score_column = 'rf_importance'
    elif method == 'permutation' and 'perm_importance' in importance_df.columns:
        score_column = 'perm_importance'
    elif method == 'shap' and 'shap_importance' in importance_df.columns:
        score_column = 'shap_importance'
    else:
        # Если указанный метод недоступен, используем первый доступный
        metric_cols = [col for col in importance_df.columns 
                      if col.endswith('_importance') or col.endswith('_score')]
        if not metric_cols:
            print("Ошибка: Не найдены столбцы с метриками важности признаков")
            return None
        score_column = metric_cols[0]
        print(f"Используем '{score_column}' для оценки важности признаков")
    
    # Сортировка по выбранной метрике
    sorted_importance = importance_df.sort_values(score_column, ascending=False)
    
    # Выбор признаков с важностью выше порога
    important_features = sorted_importance[sorted_importance[score_column] >= min_importance]
    
    # Ограничение количества признаков
    important_features = important_features.head(max_features)
    
    # Проверка наличия признаков в исходном датасете
    selected_features = [f for f in important_features['feature'].tolist() if f in df_with_features.columns]
    
    print(f"Отобрано {len(selected_features)} признаков по важности")
    
    return selected_features

def prepare_final_loyalty_dataset(df, balance_categories=True, balance_method='boundary', 
                          feature_selection=True, min_importance=0.05, max_features=30,
                          train_test_split_ratio=0.2, random_state=42, output_dir=None):
    """
    Подготовка итогового датасета с улучшенными признаками лояльности
    
    Аргументы:
        df (pandas.DataFrame): Датасет с расширенными признаками и улучшенным показателем лояльности
        balance_categories (bool): Выполнять ли балансировку категорий лояльности
        balance_method (str): Метод балансировки ('boundary', 'sampling', 'combine', 'none')
        feature_selection (bool): Выполнять ли отбор информативных признаков
        min_importance (float): Минимальный порог важности для выбора признака
        max_features (int): Максимальное количество признаков для выбора
        train_test_split_ratio (float): Доля тестовой выборки при разделении данных
        random_state (int): Случайное зерно для воспроизводимости
        output_dir (str): Директория для сохранения результатов
        
    Возвращает:
        dict: Словарь с подготовленными данными и метаданными
    """
    print("Подготовка итогового датасета с улучшенными признаками лояльности...")
    
    # Копирование датасета
    final_df = df.copy()
    
    # Проверка наличия категорий лояльности
    category_column = 'enhanced_loyalty_score_category'
    if category_column not in final_df.columns:
        print(f"Предупреждение: столбец {category_column} не найден в датасете.")
        print("Выполняем категоризацию лояльности...")
        
        if 'enhanced_loyalty_score' in final_df.columns:
            final_df, _ = categorize_loyalty_score(final_df, method='quantile', n_categories=5)
        else:
            print("Ошибка: отсутствует показатель лояльности (enhanced_loyalty_score)")
            return final_df
    
    # Создание новой целевой переменной для задачи классификации
    if category_column in final_df.columns:
        # Преобразуем категории в числовые метки, начиная с 0
        loyalty_map = {
            'Низколояльные': 0,
            'Умеренно лояльные': 1,
            'Лояльные': 2,
            'Высоколояльные': 3,
            'Отток': 0  # Объединяем с низколояльными
        }
        final_df['loyalty_target'] = final_df[category_column].map(loyalty_map)
        
        # Выводим распределение целевой переменной
        print("Распределение целевой переменной:")
        target_counts = final_df['loyalty_target'].value_counts().sort_index()
        for target, count in target_counts.items():
            print(f"  Класс {target}: {count} ({count/len(final_df)*100:.2f}%)")
        
        # Оценка баланса классов
        class_evaluation = evaluate_category_distribution(final_df, 'loyalty_target')
        
        # Выявление дисбаланса
        if class_evaluation and not class_evaluation['is_balanced'] and balance_categories:
            print(f"Выполняется балансировка классов методом '{balance_method}'...")
            
            # Определение метода балансировки
            if balance_method == 'boundary' or balance_method == 'optimize_boundaries':
                # Метод оптимизации границ категорий (уже реализован)
                # Создаем сбалансированное целевое распределение
                counts = final_df['loyalty_target'].value_counts()
                min_count = counts.min()
                max_count = counts.max()
                
                if max_count / min_count > 3:  # Существенный дисбаланс
                    # Определяем целевое распределение (в процентах)
                    if len(counts) == 5:  # Для 5 классов используем более равномерное распределение
                        target_distribution = [15, 20, 25, 25, 15]  # Небольшой перевес среднего и выше-среднего классов
                    else:
                        # Для другого количества классов используем равномерное распределение
                        target_distribution = [100 / len(counts)] * len(counts)
                    
                    # Создаем новый столбец для сбалансированных категорий
                    final_df['balanced_category'] = final_df[category_column]
                    
                    # Применяем оптимизацию границ на основе целевого распределения
                    initial_boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Примерные исходные границы
                    optimized_boundaries = optimize_category_boundaries(
                        final_df, 
                        initial_boundaries,
                        target_distribution,
                        score_column='enhanced_loyalty_score'
                    )
                    
                    # Применяем новые границы для получения сбалансированных категорий
                    final_df, _ = categorize_loyalty_score(
                        final_df, 
                        method='manual', 
                        custom_boundaries=optimized_boundaries,
                        score_column='enhanced_loyalty_score',
                        n_categories=5
                    )
                    
                    # Обновляем целевую переменную
                    final_df['balanced_target'] = final_df['enhanced_loyalty_score_category'].map(loyalty_map)
                    
                    # Выводим распределение сбалансированной целевой переменной
                    print("Распределение целевой переменной после оптимизации границ:")
                    balanced_counts = final_df['balanced_target'].value_counts().sort_index()
                    for target, count in balanced_counts.items():
                        print(f"  Класс {target}: {count} ({count/len(final_df)*100:.2f}%)")
                
                else:
                    print("Классы достаточно сбалансированы, дополнительная балансировка не требуется.")
                    
            elif balance_method == 'sampling':
                # Используем методы сэмплирования
                if IMBLEARN_AVAILABLE:
                    # Выбираем все числовые признаки для сэмплирования
                    numeric_features = final_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    features_for_sampling = [col for col in numeric_features if col != 'loyalty_target' 
                                             and not col.startswith('Клиент') 
                                             and not col.endswith('_target')]
                    
                    # Создаем копию исходного датафрейма
                    final_df['original_index'] = final_df.index
                    
                    # Применяем комбинированный метод балансировки
                    balanced_df = balance_loyalty_classes(
                        final_df,
                        target_column='loyalty_target',
                        method='combine',  # Комбинированный метод как наиболее эффективный
                        features=features_for_sampling
                    )
                    
                    # Сохраняем оригинальный индекс для идентификации новых/существующих примеров
                    balanced_df['is_synthetic'] = ~balanced_df['original_index'].isin(final_df['original_index'])
                    
                    # Копируем результаты балансировки
                    final_df = balanced_df.copy()
                else:
                    print("Ошибка: библиотека imbalanced-learn не установлена. Метод сэмплирования недоступен.")
                    print("Используем метод оптимизации границ как запасной вариант.")
                    # Здесь можно добавить код для оптимизации границ как в случае 'boundary'
                    
            elif balance_method == 'combine':
                # Комбинированный подход: сначала оптимизируем границы, затем применяем сэмплирование
                # Сначала выполняем оптимизацию границ (аналогично методу 'boundary')
                counts = final_df['loyalty_target'].value_counts()
                if len(counts) == 5:
                    target_distribution = [15, 20, 25, 25, 15]
                else:
                    target_distribution = [100 / len(counts)] * len(counts)
                
                initial_boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                optimized_boundaries = optimize_category_boundaries(
                    final_df, 
                    initial_boundaries,
                    target_distribution,
                    score_column='enhanced_loyalty_score'
                )
                
                final_df, _ = categorize_loyalty_score(
                    final_df, 
                    method='manual', 
                    custom_boundaries=optimized_boundaries,
                    score_column='enhanced_loyalty_score',
                    n_categories=5
                )
                
                final_df['balanced_target'] = final_df['enhanced_loyalty_score_category'].map(loyalty_map)
                
                # Затем применяем сэмплирование, если доступно
                if IMBLEARN_AVAILABLE:
                    numeric_features = final_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    features_for_sampling = [col for col in numeric_features if col != 'balanced_target' 
                                             and not col.startswith('Клиент') 
                                             and not col.endswith('_target')]
                    
                    final_df['original_index'] = final_df.index
                    
                    # Используем более мягкое сэмплирование, так как уже выполнили оптимизацию границ
                    balanced_df = balance_loyalty_classes(
                        final_df,
                        target_column='balanced_target',
                        method='smote',  # SMOTE для генерации синтетических примеров
                        sampling_strategy='auto',
                        features=features_for_sampling
                    )
                    
                    balanced_df['is_synthetic'] = ~balanced_df['original_index'].isin(final_df['original_index'])
                    final_df = balanced_df.copy()
                else:
                    print("Предупреждение: библиотека imbalanced-learn не установлена.")
                    print("Выполнена только оптимизация границ категорий.")
            
            elif balance_method == 'none':
                print("Балансировка классов отключена.")
            
            else:
                print(f"Предупреждение: неизвестный метод балансировки '{balance_method}'. Используем метод 'boundary' по умолчанию.")
                # Здесь можно добавить код для оптимизации границ как в случае 'boundary'
    
    # Оценка информативности признаков
    if feature_selection:
        print("Оценка информативности признаков...")
        
        # Определяем признаки, которые нужно исключить перед оценкой важности и обучением
        # Это идентификаторы, прямые производные целевой переменной или сильно скоррелированные с ней
        features_to_exclude_from_analysis = [
            'Клиент',  # ID клиента
            'enhanced_loyalty_score', # Прямой источник для enhanced_loyalty_score_category -> loyalty_target
            'rfm_segment', # Категориальный RFM, может сильно коррелировать или дублировать
            'cluster_segment', # Категориальный кластер, аналогично
            'RFM_Score', # RFM счет, если category_column из него сделана
            'R', 'F', 'M', # Базовые компоненты RFM, если они напрямую участвуют в целевой
            'weighted_RFM', # Взвешенный RFM
            'rfm_normalized', # Нормализованный RFM
            category_column, # Сам исходный категориальный столбец лояльности
            'loyalty_target' # Целевая переменная, которую мы пытаемся предсказать
        ]
        
        # Убираем дубликаты и те колонки, которых может не быть в final_df
        features_to_exclude_from_analysis = list(set(col for col in features_to_exclude_from_analysis if col in final_df.columns))

        # Создаем датафрейм X_for_importance, исключая целевую и "запрещенные" признаки
        X_for_importance = final_df.drop(columns=features_to_exclude_from_analysis, errors='ignore')
        y_for_importance = final_df['loyalty_target'] # Используем финальную целевую переменную
        
        # Убедимся, что X_for_importance содержит только числовые типы данных
        X_for_importance = X_for_importance.select_dtypes(include=np.number)

        # Проверка, что X_for_importance не пуст и содержит признаки
        if X_for_importance.empty or X_for_importance.shape[1] == 0:
            print("Ошибка: после исключения колонок не осталось признаков для оценки важности.")
            # В этом случае можно вернуть все доступные признаки, кроме целевой
            X = final_df.drop(columns=features_to_exclude_from_analysis, errors='ignore').copy()
            # y = final_df['loyalty_target'].copy() # y уже есть как y_for_importance
            if 'loyalty_target' in X.columns: # Дополнительная проверка
                 X = X.drop(columns=['loyalty_target'], errors='ignore')
            top_features = list(X.columns)
            importance_df = None # Если нет признаков для оценки, importance_df не будет создан
        else:
            importance_df = evaluate_feature_importance(
                X_features=X_for_importance, 
                y_target=y_for_importance, 
                methods=['correlation', 'mutual_info', 'rf_importance', 'permutation', 'shap'], # 'mutual_info' - было в коде, восстановил
                n_top_features=max_features, 
                random_state=random_state,
                output_dir=os.path.join(output_dir, 'feature_importance') if output_dir else None
            )
            
            # Отбор признаков на основе их важности
            # select_important_features теперь ожидает df_with_features, importance_df
            # df_with_features может быть X_for_importance, т.к. он содержит все фичи, из которых выбираем
            top_features = select_important_features( # Изменил распаковку, т.к. функция возвращает только список
                df_with_features=X_for_importance, 
                importance_df=importance_df, 
                min_importance=min_importance, 
                max_features=max_features
            )
            if top_features is None: # Если select_important_features вернул None (например, importance_df был None)
                print("Не удалось отобрать признаки, используем все доступные из X_for_importance.")
                top_features = X_for_importance.columns.tolist()

            
            # Формируем X и y на основе отобранных признаков
            if top_features: # Убедимся, что top_features не пуст
                X = final_df[top_features].copy() 
            else: # Если top_features пуст, но X_for_importance не был, берем все из X_for_importance
                X = X_for_importance.copy()
                top_features = X.columns.tolist()

            y = final_df['loyalty_target'].copy()
            
            # ЕЩЕ РАЗ УБЕДИМСЯ, что никакие из запрещенных признаков не попали в X случайно
            # Это должно быть не нужно, если top_features корректно отобраны из X_for_importance
            final_X_cols_to_drop = [col for col in features_to_exclude_from_analysis if col in X.columns and col != 'loyalty_target']
            if final_X_cols_to_drop:
                X.drop(columns=final_X_cols_to_drop, inplace=True)
                print(f"Дополнительно удалены из X: {final_X_cols_to_drop}")
                top_features = list(X.columns) # Обновляем top_features

    else:
        # Если отбор признаков не выполняется, берем все, кроме целевой и запрещенных
        features_to_exclude = [
            'Клиент', category_column, 'loyalty_target', 'enhanced_loyalty_score',
            'rfm_segment', 'cluster_segment', 'RFM_Score', 'R', 'F', 'M',
            'weighted_RFM', 'rfm_normalized'
        ]
        features_to_exclude = list(set(col for col in features_to_exclude if col in final_df.columns))
        
        X = final_df.drop(columns=features_to_exclude, errors='ignore').copy()
        y = final_df['loyalty_target'].copy()
        top_features = list(X.columns)
        print(f"Отбор признаков не производился. Используется {len(top_features)} признаков.")

    print(f"Итоговое количество признаков для моделирования: {X.shape[1]}")
    if X.shape[1] < 5 and feature_selection:
        print(f"ПРЕДУПРЕЖДЕНИЕ: отобрано очень мало признаков ({X.shape[1]}). Проверьте пороги отбора.")
    print(f"Признаки, используемые для обучения: {list(X.columns)}")

    # Разделение данных на обучающую и тестовую выборки
    print(f"Разделение на обучающую и тестовую выборки (тест: {train_test_split_ratio*100}%)...")
    
    # Определяем, какую целевую переменную использовать для стратификации
    # и как она будет называться в сохраненных файлах и метаданных.
    # y_for_split - это Series, который будет разделен на y_train и y_test.
    # target_column_name - имя этой целевой переменной.
    
    # y у нас уже определен выше как final_df['loyalty_target'].copy()
    # X также определен выше и содержит финальный набор признаков.
    
    if 'balanced_target' in final_df.columns and balance_categories and balance_method != 'none':
        # Если была балансировка, которая создала 'balanced_target', и она не была отключена,
        # и эта колонка действительно существует в final_df.
        y_to_stratify_and_learn = final_df['balanced_target'].copy()
        target_column_for_split_and_output = 'balanced_target'
        print(f"Целевая переменная для обучения и стратификации: {target_column_for_split_and_output}")
    else:
        y_to_stratify_and_learn = y.copy() # y это final_df['loyalty_target'].copy()
        target_column_for_split_and_output = 'loyalty_target'
        print(f"Целевая переменная для обучения и стратификации: {target_column_for_split_and_output}")

    # Разделяем X (который уже готов) и y_to_stratify_and_learn
    # Важно: если X был результатом сэмплинга, то y_to_stratify_and_learn должен соответствовать ему.
    # На данный момент логика балансировки (если sampling) возвращает новый X и y.
    # Если балансировка была 'boundary', то X и y остаются оригинальными срезами из final_df.
    # Это нужно учесть: если X и y пришли из balance_loyalty_classes, они уже сбалансированы.
    # В таком случае y_to_stratify_and_learn должен быть этим новым y из сэмплинга.
    
    # ПРОВЕРКА: Если X и y были изменены функцией balance_loyalty_classes (методом sampling/combine),
    # то y_to_stratify_and_learn должен быть этим новым y. 
    # На данный момент X и y формируются до блока балансировки классов, а потом если sampling, то X и y могут быть переопределены
    # Этот участок кода выполняется ПОСЛЕ блока балансировки.
    # Если final_df был заменен на balanced_df, то X и y, созданные из него, уже сбалансированы.
    # Однако, если использовался метод 'sampling' в balance_categories, то `final_df` перезаписывался:
    # `final_df = balanced_df.copy()` где balanced_df результат balance_loyalty_classes.
    # Тогда X и y, которые создавались ВЫШЕ из final_df, уже будут из сбалансированных данных.
    # В таком случае, y_to_stratify_and_learn, взятый из этого final_df, тоже будет сбалансированным.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_to_stratify_and_learn, test_size=train_test_split_ratio, random_state=random_state, stratify=y_to_stratify_and_learn
    )
    
    print(f"Размер обучающей выборки: {len(X_train)} примеров")
    print(f"Размер тестовой выборки: {len(X_test)} примеров")
    
    # Проверка распределения классов в выборках
    train_dist = y_train.value_counts(normalize=True) * 100
    test_dist = y_test.value_counts(normalize=True) * 100
    
    print("Распределение классов в обучающей выборке:")
    for cls, pct in train_dist.sort_index().items():
        print(f"  Класс {cls}: {pct:.2f}%")
    
    print("Распределение классов в тестовой выборке:")
    for cls, pct in test_dist.sort_index().items():
        print(f"  Класс {cls}: {pct:.2f}%")
    
    # Сохранение итогового датасета и разделенных выборок
    if output_dir:
        # Создаем директорию для данных моделирования
        model_dir = os.path.join(output_dir, 'model_data')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Сохраняем данные, которые пошли в обучение
        # Это X_train, y_train, X_test, y_test
        # Имя целевого столбца берем из target_column_for_split_and_output
        
        X_train.to_csv(os.path.join(model_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(model_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(model_dir, 'y_train.csv'), index=False, header=[target_column_for_split_and_output])
        y_test.to_csv(os.path.join(model_dir, 'y_test.csv'), index=False, header=[target_column_for_split_and_output])
        
        print(f"Обучающие и тестовые данные сохранены в {model_dir}")

        # Сохраняем метаданные
        metadata = {
            'feature_count': X.shape[1],
            'feature_list': list(X.columns),
            'target_column': target_column_for_split_and_output, # Это правильная переменная
            'train_size': len(X_train),
            'test_size': len(X_test),
            'class_distribution': y_to_stratify_and_learn.value_counts().to_dict(), # Используем y_to_stratify_and_learn
            'train_distribution': y_train.value_counts().to_dict(),
            'test_distribution': y_test.value_counts().to_dict(),
            'balance_method': balance_method if balance_categories else 'none',
            'feature_selection': feature_selection,
            'min_importance': min_importance,
            'max_features': max_features,
            'train_test_split_ratio': train_test_split_ratio,
            'random_state': random_state
        }
        
        import json
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Обучающие и тестовые выборки сохранены в директории {model_dir}")
    
    # Вывод информации о готовом датасете
    print(f"Подготовка итогового датасета завершена.")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Размер датасета (X) перед train/test split: {len(X)} примеров") # Изменено для ясности
    
    # Получение фактических имен классов из данных
    if 'enhanced_loyalty_score_category' in final_df.columns:
        class_names = final_df['enhanced_loyalty_score_category'].unique().tolist()
        class_names.sort()  # Сортируем для стабильного порядка
    else:
        # Используем стандартные имена классов как запасной вариант
        class_names = ['Отток', 'Низколояльные', 'Умеренно лояльные', 'Лояльные', 'Высоколояльные']
    
    # Возвращаем словарь с данными и метаданными
    result = {
        'full_dataset': final_df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'features': X.columns.tolist(),
        'target_column': target_column_for_split_and_output,
        'class_names': class_names
    }
    
    return result

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