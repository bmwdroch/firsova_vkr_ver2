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
from preprocessing.data_preprocessing import preprocess_data, prepare_final_dataset
from preprocessing.enhanced_loyalty_features import (
    create_enhanced_features, 
    perform_customer_clustering,
    calculate_enhanced_loyalty_score,
    prepare_final_loyalty_dataset,
    get_category_profiles
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
    if 'enhanced_loyalty_score_category' in enhanced_df.columns:
        order = ['Высоколояльные', 'Лояльные', 'Умеренно лояльные', 'Низколояльные', 'Отток']
        counts = enhanced_df['enhanced_loyalty_score_category'].value_counts().reindex(order)
        plt.bar(counts.index, counts.values)
        plt.title('Улучшенный метод')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'loyalty_distribution_comparison.png'))
    plt.close()
    
    # 2. Матрица сравнения категорий
    if 'loyalty_segment' in base_df.columns and 'enhanced_loyalty_score_category' in enhanced_df.columns:
        # Объединяем таблицы для сравнения
        merged_df = base_df[['Клиент', 'loyalty_segment']].merge(
            enhanced_df[['Клиент', 'enhanced_loyalty_score_category']], 
            on='Клиент', 
            how='inner'
        )
        
        # Создаем матрицу сравнения
        plt.figure(figsize=(10, 8))
        
        order = ['Высоколояльные', 'Лояльные', 'Умеренно лояльные', 'Низколояльные', 'Отток']
        comparison = pd.crosstab(
            merged_df['loyalty_segment'], 
            merged_df['enhanced_loyalty_score_category'],
            normalize='index'
        ).reindex(index=order, columns=order)
        
        sns.heatmap(comparison, annot=True, fmt='.2f', cmap='YlGnBu')
        plt.xlabel('Улучшенный метод')
        plt.ylabel('Базовый RFM-метод')
        plt.title('Матрица переходов между категориями лояльности')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'loyalty_confusion_matrix.png'))
        plt.close()
    
    # 3. Сравнение профилей категорий
    if 'enhanced_loyalty_score_category' in enhanced_df.columns:
        # Получение профилей категорий
        category_profiles = get_category_profiles(enhanced_df, 'enhanced_loyalty_score_category')
        
        if category_profiles is not None:
            # Визуализация ключевых метрик по категориям
            selected_metrics = []
            
            # Выбор наиболее важных метрик для отображения (максимум 8)
            metric_candidates = ['recency', 'frequency', 'monetary', 'avg_purchase',
                                'bonus_usage_ratio', 'product_diversity_gini', 
                                'purchase_stability', 'recency_ratio', 'location_loyalty']
            
            for metric in metric_candidates:
                if metric in category_profiles.columns:
                    selected_metrics.append(metric)
                    if len(selected_metrics) >= 8:
                        break
            
            if selected_metrics:
                # Нормализация метрик для удобства отображения
                profile_norm = category_profiles[selected_metrics].copy()
                for col in selected_metrics:
                    profile_norm[col] = (profile_norm[col] - profile_norm[col].min()) / \
                                        (profile_norm[col].max() - profile_norm[col].min())
                
                # Визуализация в виде тепловой карты
                plt.figure(figsize=(12, 6))
                sns.heatmap(profile_norm.T, annot=True, fmt='.2f', cmap='viridis', 
                            linewidths=0.5, cbar_kws={'label': 'Нормализованные значения'})
                plt.title('Профили категорий лояльности')
                plt.ylabel('Метрики')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'loyalty_category_profiles.png'))
                plt.close()
                
                # Радарная диаграмма для категорий
                categories = profile_norm.index.tolist()
                num_metrics = len(selected_metrics)
                
                # Преобразуем метрики для отображения на радарной диаграмме
                angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
                angles += angles[:1]  # Замыкаем круг
                
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                
                for i, category in enumerate(categories):
                    values = profile_norm.loc[category, selected_metrics].tolist()
                    values += values[:1]  # Замыкаем круг
                    ax.plot(angles, values, linewidth=2, label=category)
                    ax.fill(angles, values, alpha=0.1)
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(selected_metrics)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
                ax.grid(True)
                
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                plt.title('Радарная диаграмма профилей категорий лояльности')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'loyalty_radar_profiles.png'))
                plt.close()
    
    # 4. Распределение показателя лояльности с учётом категорий
    if 'enhanced_loyalty_score' in enhanced_df.columns and 'enhanced_loyalty_score_category' in enhanced_df.columns:
        plt.figure(figsize=(12, 6))
        
        # Создаем KDE-график для каждой категории
        order = ['Высоколояльные', 'Лояльные', 'Умеренно лояльные', 'Низколояльные', 'Отток']
        categories = [cat for cat in order if cat in enhanced_df['enhanced_loyalty_score_category'].unique()]
        
        for category in categories:
            category_data = enhanced_df[enhanced_df['enhanced_loyalty_score_category'] == category]['enhanced_loyalty_score']
            sns.kdeplot(category_data, label=category, fill=True, alpha=0.3)
        
        plt.title('Распределение показателя лояльности по категориям')
        plt.xlabel('Показатель лояльности')
        plt.ylabel('Плотность')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'loyalty_score_by_category.png'))
        plt.close()
        
    # 5. Сравнение распределения до и после балансировки (если есть данные)
    if 'enhanced_loyalty_score_category' in enhanced_df.columns and 'balanced_category' in enhanced_df.columns:
        plt.figure(figsize=(12, 6))
        
        # Распределение до балансировки
        plt.subplot(1, 2, 1)
        before_counts = enhanced_df['enhanced_loyalty_score_category'].value_counts().sort_index()
        plt.bar(before_counts.index, before_counts.values)
        plt.title('До балансировки')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Количество клиентов')
        
        # Распределение после балансировки
        plt.subplot(1, 2, 2)
        after_counts = enhanced_df['balanced_category'].value_counts().sort_index()
        plt.bar(after_counts.index, after_counts.values)
        plt.title('После балансировки')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'loyalty_balance_comparison.png'))
        plt.close()
        
    # 6. Визуализация важности признаков (если есть файл)
    feature_importance_path = os.path.join(output_dir, 'feature_importance.csv')
    if os.path.exists(feature_importance_path):
        try:
            # Загружаем данные о важности признаков
            importance_df = pd.read_csv(feature_importance_path)
            
            if 'importance_score' in importance_df.columns and 'feature' in importance_df.columns:
                # Визуализация топ-15 признаков
                top_n = min(15, len(importance_df))
                top_features = importance_df.sort_values('importance_score', ascending=False).head(top_n)
                
                plt.figure(figsize=(12, 8))
                g = sns.barplot(x='importance_score', y='feature', data=top_features, palette='viridis')
                plt.title(f'Топ-{top_n} наиболее информативных признаков')
                plt.xlabel('Оценка важности')
                plt.ylabel('Признак')
                
                # Добавляем значения на график
                for i, v in enumerate(top_features['importance_score']):
                    g.text(v + 0.01, i, f'{v:.3f}', va='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'top_features.png'))
                plt.close()
                
                # Если есть другие метрики важности, визуализируем их
                importance_metrics = [col for col in importance_df.columns 
                                     if col.endswith('_importance') or col.endswith('_norm')]
                
                if len(importance_metrics) > 1:
                    # Визуализация сравнения методов оценки важности (топ-10)
                    top_10 = importance_df.sort_values('importance_score', ascending=False).head(10)
                    
                    # Выбираем метрики для сравнения
                    compare_metrics = [m for m in importance_metrics if m != 'importance_score'][:3]
                    
                    if compare_metrics:
                        for metric in compare_metrics:
                            plt.figure(figsize=(10, 6))
                            top_by_metric = top_10.sort_values(metric, ascending=False)
                            sns.barplot(x=metric, y='feature', data=top_by_metric)
                            metric_name = metric.replace('_importance', '').replace('_norm', '').capitalize()
                            plt.title(f'Топ-10 признаков по {metric_name}')
                            plt.tight_layout()
                            plt.savefig(os.path.join(vis_dir, f'top_features_{metric}.png'))
                            plt.close()
                
                # Визуализация распределения важности признаков
                plt.figure(figsize=(10, 6))
                sns.histplot(importance_df['importance_score'], bins=20, kde=True)
                plt.title('Распределение важности признаков')
                plt.xlabel('Важность')
                plt.ylabel('Количество признаков')
                plt.axvline(x=0.05, color='r', linestyle='--', alpha=0.7, label='Порог 0.05')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'feature_importance_distribution.png'))
                plt.close()
        except Exception as e:
            print(f"Ошибка при визуализации важности признаков: {e}")
    
    # 7. Визуализация сравнения тренировочной и тестовой выборок
    model_data_dir = os.path.join(output_dir, 'model_data')
    metadata_path = os.path.join(model_data_dir, 'metadata.json')
    
    if os.path.exists(metadata_path):
        try:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Визуализация распределения классов в тренировочной и тестовой выборках
            if 'train_distribution' in metadata and 'test_distribution' in metadata:
                train_dist = metadata['train_distribution']
                test_dist = metadata['test_distribution']
                
                # Преобразуем в датафрейм для удобства визуализации
                distribution_data = []
                
                for cls in sorted(train_dist.keys()):
                    distribution_data.append({
                        'Class': cls, 
                        'Split': 'Train', 
                        'Percentage': train_dist[cls] / sum(train_dist.values()) * 100
                    })
                    distribution_data.append({
                        'Class': cls, 
                        'Split': 'Test', 
                        'Percentage': test_dist[cls] / sum(test_dist.values()) * 100
                    })
                
                dist_df = pd.DataFrame(distribution_data)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Class', y='Percentage', hue='Split', data=dist_df)
                plt.title('Распределение классов в тренировочной и тестовой выборках')
                plt.xlabel('Класс')
                plt.ylabel('Процент')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'train_test_distribution.png'))
                plt.close()
                
                # Визуализация метаданных выборок
                plt.figure(figsize=(10, 6))
                sizes = [metadata['train_size'], metadata['test_size']]
                labels = [f'Обучающая ({sizes[0]} примеров)', f'Тестовая ({sizes[1]} примеров)']
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#99ff99'])
                plt.title(f'Разделение на обучающую и тестовую выборки ({metadata.get("train_test_split_ratio", 0.2)*100}%)')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'train_test_split.png'))
                plt.close()
        except Exception as e:
            print(f"Ошибка при визуализации метаданных выборок: {e}")
    
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
    parser.add_argument('--balance_method', type=str, default='boundary', 
                       choices=['boundary', 'sampling', 'combine', 'none'], 
                       help='Метод балансировки классов')
    parser.add_argument('--no_balance', action='store_true', help='Отключить балансировку классов')
    parser.add_argument('--no_feature_selection', action='store_true', help='Отключить отбор признаков')
    parser.add_argument('--min_importance', type=float, default=0.05, help='Минимальный порог важности признака')
    parser.add_argument('--max_features', type=int, default=30, help='Максимальное количество отбираемых признаков')
    parser.add_argument('--test_size', type=float, default=0.2, help='Доля тестовой выборки при разделении данных')
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
    
    # Проверка необходимости балансировки
    balance_categories = not args.no_balance
    
    # Проверка необходимости отбора признаков
    feature_selection = not args.no_feature_selection
    
    # Подготовка итогового датасета с выбранным методом балансировки и отбором признаков
    result = prepare_final_loyalty_dataset(
        loyalty_df, 
        balance_categories=balance_categories, 
        balance_method=args.balance_method,
        feature_selection=feature_selection,
        min_importance=args.min_importance,
        max_features=args.max_features,
        train_test_split_ratio=args.test_size,
        output_dir=args.output
    )
    
    # Получаем полный датасет из результата
    if isinstance(result, dict) and 'full_dataset' in result:
        final_df = result['full_dataset']
    else:
        final_df = result  # Для обратной совместимости
    
    # Визуализация сравнения методов
    visualize_comparison(base_df, final_df, args.output)
    
    print("Демонстрация расширенной оценки лояльности клиентов завершена успешно.")
    print(f"Результаты сохранены в директории {args.output}")
    
    # Дополнительная информация о балансировке
    if balance_categories:
        print(f"Метод балансировки: {args.balance_method}")
        # Если есть сбалансированная целевая переменная, показываем статистику
        if 'balanced_target' in final_df.columns:
            balanced_counts = final_df['balanced_target'].value_counts().sort_index()
            print("Распределение сбалансированной целевой переменной:")
            for target, count in balanced_counts.items():
                print(f"  Класс {target}: {count} ({count/len(final_df)*100:.2f}%)")
        
        # Если есть синтетические примеры, показываем их количество
        if 'is_synthetic' in final_df.columns:
            synthetic_count = final_df['is_synthetic'].sum()
            if synthetic_count > 0:
                print(f"Создано {synthetic_count} синтетических примеров ({synthetic_count/len(final_df)*100:.2f}% от общего)")
    else:
        print("Балансировка классов отключена.")
    
    # Информация об отборе признаков
    if feature_selection:
        print(f"Выполнен отбор признаков (мин. важность: {args.min_importance}, макс. количество: {args.max_features})")
        if isinstance(result, dict) and 'features' in result:
            print(f"Отобрано {len(result['features'])} информативных признаков")
    else:
        print("Отбор признаков отключен.")
        
    # Сводная информация о структуре итогового датасета
    print("\nСтруктура итогового датасета:")
    
    if isinstance(result, dict):
        print(f"Количество клиентов: {len(result['full_dataset'])}")
        print(f"Количество признаков: {len(result['features'])}")
        print(f"Целевая переменная: {result['target_column']}")
        print(f"Размер обучающей выборки: {len(result['X_train'])} примеров")
        print(f"Размер тестовой выборки: {len(result['X_test'])} примеров")
    else:
        print(f"Количество клиентов: {len(final_df)}")
        print(f"Количество признаков: {len(final_df.columns)}")
    
    print("Готово!")

if __name__ == "__main__":
    main() 