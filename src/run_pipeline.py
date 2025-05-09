#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Основной скрипт для запуска полного пайплайна анализа и обучения моделей.
Включает предобработку данных, создание признаков лояльности и обучение моделей.
"""

import os
import sys
import argparse
import pandas as pd
import pickle
from datetime import datetime
import warnings
import numpy as np

# Подавление предупреждений
warnings.filterwarnings('ignore')

# Импорт модулей проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing.data_preprocessing import preprocess_data, prepare_final_dataset
from preprocessing.enhanced_loyalty_features import (
    create_enhanced_features,
    perform_customer_clustering,
    calculate_enhanced_loyalty_score,
    prepare_final_loyalty_dataset
)
from modeling.model_training import ModelTrainer
from modeling.model_evaluation import ModelEvaluator


def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        argparse.Namespace: Аргументы командной строки.
    """
    parser = argparse.ArgumentParser(description='Полный пайплайн для анализа лояльности клиентов')
    
    parser.add_argument('--input_file', type=str, default='../dataset/Concept202408.csv',
                       help='Путь к исходному CSV-файлу')
    parser.add_argument('--output_dir', type=str, default='../output',
                       help='Директория для сохранения результатов')
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Пропустить этап предобработки (использовать существующие данные)')
    parser.add_argument('--skip_loyalty_features', action='store_true',
                       help='Пропустить этап создания признаков лояльности (использовать существующие данные)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Размер тестовой выборки')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Seed для генератора случайных чисел')
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='Выполнить настройку гиперпараметров моделей')
    parser.add_argument('--create_ensemble', action='store_true',
                       help='Создать ансамблевые модели')
    parser.add_argument('--balance_classes', action='store_true',
                       help='Применить балансировку классов при обучении моделей')
    parser.add_argument('--save_intermediate', action='store_true',
                       help='Сохранять промежуточные результаты')
    parser.add_argument('--skip_model_training', action='store_true',
                       help='Пропустить этап обучения моделей (использовать сохраненные)')
    
    return parser.parse_args()


def preprocess_data_step(input_file, output_dir, save_intermediate=False):
    """
    Выполнение шага предобработки данных.
    
    Args:
        input_file (str): Путь к исходному CSV-файлу.
        output_dir (str): Директория для сохранения результатов.
        save_intermediate (bool): Сохранять ли промежуточные результаты.
        
    Returns:
        pandas.DataFrame: Предобработанный датасет.
    """
    print("\n--- Шаг 1: Предобработка данных ---")
    
    # Проверка существования файла
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Входной файл не найден: {input_file}")
    
    # Создание директории для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Загрузка данных
    print(f"Загрузка данных из {input_file}...")
    raw_df = pd.read_csv(input_file)
    print(f"Данные загружены. Размерность: {raw_df.shape}")
    
    # Предобработка данных
    print("Выполнение предобработки данных...")
    preprocessed_df = preprocess_data(raw_df, output_dir)
    
    # Подготовка базового датасета с RFM
    print("Создание базового RFM-анализа...")
    base_df = prepare_final_dataset(preprocessed_df)
    
    # Сохранение промежуточных результатов
    if save_intermediate:
        base_output_path = os.path.join(output_dir, 'base_rfm_dataset.pkl')
        with open(base_output_path, 'wb') as f:
            pickle.dump(base_df, f)
        print(f"Базовый RFM-датасет сохранен в {base_output_path}")
    
    return base_df


def create_loyalty_features_step(base_df, output_dir, save_intermediate=False):
    """
    Выполнение шага создания признаков лояльности.
    
    Args:
        base_df (pandas.DataFrame): Базовый датасет с RFM-анализом.
        output_dir (str): Директория для сохранения результатов.
        save_intermediate (bool): Сохранять ли промежуточные результаты.
        
    Returns:
        dict: Словарь с подготовленными данными для моделирования.
    """
    print("\n--- Шаг 2: Создание признаков лояльности ---")
    
    # Создание директории для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Создание расширенных признаков
    print("Создание расширенных признаков лояльности...")
    enhanced_df = create_enhanced_features(base_df)
    
    # Выполнение кластеризации клиентов
    print("Выполнение кластеризации клиентов...")
    clustered_df = perform_customer_clustering(enhanced_df)
    
    # Расчет улучшенного показателя лояльности
    print("Расчет улучшенного показателя лояльности...")
    loyalty_df = calculate_enhanced_loyalty_score(clustered_df)
    
    # Подготовка финального датасета для моделирования
    print("Подготовка финального датасета для моделирования...")
    data_dict = prepare_final_loyalty_dataset(
        loyalty_df, 
        balance_method='none',  # Без балансировки на этом этапе
        feature_selection=True,   # Отбор информативных признаков
        max_features=50
    )
    
    # Сохранение промежуточных результатов
    if save_intermediate:
        # Сохранение датасета с признаком лояльности
        loyalty_output_path = os.path.join(output_dir, 'loyalty_features_dataset.pkl')
        with open(loyalty_output_path, 'wb') as f:
            pickle.dump(loyalty_df, f)
        print(f"Датасет с признаками лояльности сохранен в {loyalty_output_path}")
    
    # Сохранение финального датасета для моделирования
    dataset_output_path = os.path.join(output_dir, 'loyalty_dataset.pkl')
    with open(dataset_output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Финальный датасет для моделирования сохранен в {dataset_output_path}")
    
    return data_dict


def train_models_step(data_dict, output_dir, tune_hyperparams=False, create_ensemble=False,
                     balance_classes=False, random_state=42, skip_model_training=False, save_intermediate=False):
    """
    Выполнение шага обучения моделей.
    
    Args:
        data_dict (dict): Словарь с данными для моделирования.
        output_dir (str): Директория для сохранения результатов.
        tune_hyperparams (bool): Выполнять ли настройку гиперпараметров.
        create_ensemble (bool): Создавать ли ансамблевые модели.
        balance_classes (bool): Применять ли балансировку классов.
        random_state (int): Seed для генератора случайных чисел.
        skip_model_training (bool): Пропускать ли обучение моделей.
        save_intermediate (bool): Сохранять ли промежуточные модели и метрики.
        
    Returns:
        tuple: (ModelTrainer, ModelEvaluator) - объекты для обучения и оценки моделей.
    """
    print("\n--- Шаг 3: Обучение моделей классификации ---")
    
    # Создание директорий для результатов
    models_dir = os.path.join(output_dir, 'models')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    trained_models_path = os.path.join(models_dir, 'trained_models_bundle.pkl')
    
    # Извлечение данных из словаря
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_names = data_dict.get('features', [])
    class_names = data_dict.get('class_names', [])

    # Преобразование в numpy array
    X_train = X_train.to_numpy() if hasattr(X_train, 'to_numpy') else np.array(X_train)
    X_test = X_test.to_numpy() if hasattr(X_test, 'to_numpy') else np.array(X_test)
    y_train = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else np.array(y_train)
    y_test = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else np.array(y_test)

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Количество признаков: {X_train.shape[1]}")
    
    # Создаем маппинг числовых меток на имена классов
    class_mapping = {i: name for i, name in enumerate(class_names)} if class_names else {i: str(i) for i in range(len(np.unique(y_train)))}
    
    print(f"Распределение классов (обучающая выборка):")
    unique_classes = np.unique(y_train)
    for class_idx in sorted(unique_classes):
        count = np.sum(y_train == class_idx)
        class_name = class_mapping.get(class_idx, f"Класс {class_idx}")
        print(f"  - {class_name}: {count} ({count/len(y_train):.2%})")
    
    # Создание объекта для обучения моделей
    trainer = ModelTrainer(models_dir=models_dir, results_dir=results_dir)
    evaluator = ModelEvaluator(trainer.models, results_dir=results_dir) # Инициализируем здесь
    all_metrics = {}

    if skip_model_training and os.path.exists(trained_models_path):
        print(f"Загрузка ранее обученных моделей и метрик из {trained_models_path}...")
        with open(trained_models_path, 'rb') as f:
            saved_bundle = pickle.load(f)
            trainer.models = saved_bundle['models']
            all_metrics = saved_bundle['metrics']
            # Обновляем evaluator новым словарем моделей
            evaluator.models = trainer.models 
        print("Модели и метрики загружены.")
    else:
        if skip_model_training:
            print(f"Файл {trained_models_path} не найден. Модели будут обучены заново.")

        # Определение весов классов для учета несбалансированности
        if balance_classes:
            print("Применение весов классов для учета несбалансированности...")
            class_counts = np.bincount(y_train)
            class_weights = {i: len(y_train) / (len(unique_classes) * count) 
                             for i, count in enumerate(class_counts)}
        else:
            class_weights = None
        
        # Добавление базовых моделей
        print("Добавление базовых моделей...")
        trainer.add_base_models(class_weights=class_weights)
        
        # Обучение всех моделей
        print("Обучение моделей...")
        trainer.train_all_models(X_train, y_train)
        
        # Настройка гиперпараметров (если требуется)
        if tune_hyperparams:
            print("Настройка гиперпараметров...")
            best_models = trainer.tune_all_models(
                X_train, y_train, 
                cv=5, 
                method='random', 
                n_iter=20
            )
            for name, model in best_models.items():
                trainer.models[name] = model
        
        # Создание ансамблевых моделей (если требуется)
        if create_ensemble:
            print("Создание ансамблевых моделей...")
            models_for_ensemble = [(name, model) for name, model in trainer.models.items() 
                                  if name not in ['voting_ensemble', 'stacking_ensemble']]
            
            if len(models_for_ensemble) > 1:
                print("Создание ансамбля голосования...")
                trainer.create_voting_ensemble(estimators=models_for_ensemble, voting='soft')
                trainer.train_model('voting_ensemble', X_train, y_train)
                
                print("Создание стекинг-ансамбля...")
                trainer.create_stacking_ensemble(estimators=models_for_ensemble)
                trainer.train_model('stacking_ensemble', X_train, y_train)
        
        # Оценка всех моделей на тестовой выборке
        print("Оценка моделей на тестовой выборке...")
        all_metrics = evaluator.evaluate_models(
            X_test, y_test, class_names=class_names
        )

        if save_intermediate:
            print(f"Сохранение обученных моделей и метрик в {trained_models_path}...")
            with open(trained_models_path, 'wb') as f:
                pickle.dump({'models': trainer.models, 'metrics': all_metrics}, f)
            print("Модели и метрики сохранены.")

    # Передаем all_metrics в ModelEvaluator, если они были загружены или вычислены
    evaluator.evaluation_results = all_metrics
    
    # Создание сравнительных визуализаций
    print("Создание сравнительных визуализаций...")
    evaluator.plot_metrics_comparison(all_metrics)
    
    # Визуализация важности признаков для интерпретируемых моделей
    print("Визуализация важности признаков...")
    for model_name in ['random_forest', 'xgboost', 'lightgbm']:
        if model_name in trainer.models:
            try:
                evaluator.plot_feature_importance(
                    trainer.models[model_name], model_name,
                    X_train, feature_names,
                    top_n=20
                )
            except Exception as e:
                print(f"Ошибка при визуализации важности признаков для {model_name}: {e}")
    
    # Сохранение лучшей модели
    best_model_name = max(all_metrics, key=lambda x: all_metrics[x].get('f1_macro', 0.0))
    best_model = trainer.models[best_model_name]
    
    print(f"\nЛучшая модель по F1-macro: {best_model_name}")
    print(f"Метрики лучшей модели:")
    if best_model_name in all_metrics:
        for metric, value in all_metrics[best_model_name].items():
            if isinstance(value, (int, float)):
                print(f"  - {metric}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"  - {metric}: (dict with keys: {', '.join(value.keys())})")
            elif isinstance(value, list):
                print(f"  - {metric}: (list of length {len(value)})")
            else:
                print(f"  - {metric}: {value}")
    else:
        print(f"  Метрики для модели {best_model_name} не найдены.")
    
    # Сохранение лучшей модели в отдельный файл
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump({
            'model': best_model, 
            'name': best_model_name,
            'class_mapping': class_mapping,
            'feature_names': feature_names
        }, f)
    print(f"Лучшая модель сохранена в {best_model_path}")
    
    return trainer, evaluator


def main():
    """
    Основная функция для запуска полного пайплайна обработки данных и обучения моделей.
    """
    # Парсинг аргументов командной строки
    args = parse_args()
    
    # Установка времени начала выполнения
    start_time = datetime.now()
    print(f"Начало выполнения: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Шаг 1: Предобработка данных
    if not args.skip_preprocessing:
        try:
            base_df = preprocess_data_step(
                args.input_file, 
                args.output_dir,
                save_intermediate=args.save_intermediate
            )
        except Exception as e:
            print(f"Ошибка при предобработке данных: {e}")
            return
    else:
        print("\n--- Шаг 1: Предобработка данных пропущена ---")
        # Загрузка существующего базового датасета
        base_file = os.path.join(args.output_dir, 'base_rfm_dataset.pkl')
        if os.path.exists(base_file):
            with open(base_file, 'rb') as f:
                base_df = pickle.load(f)
            print(f"Загружен существующий базовый датасет из {base_file}")
        else:
            print(f"Ошибка: Базовый датасет не найден в {base_file}")
            return
    
    # Шаг 2: Создание признаков лояльности
    if not args.skip_loyalty_features:
        try:
            data_dict = create_loyalty_features_step(
                base_df, 
                args.output_dir,
                save_intermediate=args.save_intermediate
            )
        except Exception as e:
            print(f"Ошибка при создании признаков лояльности: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("\n--- Шаг 2: Создание признаков лояльности пропущено ---")
        # Загрузка существующего датасета для моделирования
        dataset_file = os.path.join(args.output_dir, 'loyalty_dataset.pkl')
        if os.path.exists(dataset_file):
            with open(dataset_file, 'rb') as f:
                data_dict = pickle.load(f)
            print(f"Загружен существующий датасет для моделирования из {dataset_file}")
        else:
            print(f"Ошибка: Датасет для моделирования не найден в {dataset_file}")
            return
    
    # Шаг 3 и 4: Обучение и оценка моделей
    try:
        trainer, evaluator = train_models_step(
            data_dict, 
            args.output_dir,
            tune_hyperparams=args.tune_hyperparams,
            create_ensemble=args.create_ensemble,
            balance_classes=args.balance_classes,
            random_state=args.random_state,
            skip_model_training=args.skip_model_training,
            save_intermediate=args.save_intermediate
        )
    except Exception as e:
        print(f"Ошибка при обучении моделей: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Вывод времени выполнения
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nВыполнение завершено в {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Общее время выполнения: {duration}")
    
    print("\nРезультаты сохранены в директории:")
    print(f"  - {args.output_dir}")


if __name__ == "__main__":
    main() 