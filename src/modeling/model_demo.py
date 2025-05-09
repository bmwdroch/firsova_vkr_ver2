#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Демонстрационный скрипт для обучения моделей классификации лояльности клиентов.
Включает демонстрацию основных моделей, их сравнение и визуализацию результатов.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

# Импорт собственных модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.model_training import ModelTrainer, get_hyperparameter_grids
from modeling.model_evaluation import ModelEvaluator

# Отключение предупреждений
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-whitegrid')
sns.set_palette('Set2')


def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        argparse.Namespace: Аргументы командной строки.
    """
    parser = argparse.ArgumentParser(description='Демонстрация обучения моделей классификации лояльности')
    
    parser.add_argument('--data_path', type=str, default='../output/loyalty_dataset.pkl',
                       help='Путь к подготовленному датасету')
    parser.add_argument('--output_dir', type=str, default='../output/demo',
                       help='Директория для сохранения результатов')
    parser.add_argument('--balance_classes', action='store_true',
                       help='Использовать балансировку классов (веса)')
    parser.add_argument('--use_ensemble', action='store_true',
                       help='Создать и оценить ансамблевые модели')
    parser.add_argument('--tune_models', action='store_true',
                       help='Выполнить настройку гиперпараметров моделей')
    parser.add_argument('--n_top_features', type=int, default=20,
                       help='Количество важных признаков для визуализации')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Размер тестовой выборки')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Seed для генератора случайных чисел')
    
    return parser.parse_args()


def load_dataset(data_path):
    """
    Загрузка и подготовка датасета для обучения моделей.
    
    Args:
        data_path (str): Путь к файлу с данными.
        
    Returns:
        tuple: X, y, feature_names, class_names
    """
    print(f"Загрузка данных из {data_path}...")
    
    if data_path.endswith('.pkl'):
        # Загрузка из pickle
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict) and 'X_train' in data:
            # Данные уже разделены на обучающую и тестовую выборки
            print("Обнаружены предварительно разделенные данные")
            return (data['X_train'], data['X_test'], 
                   data['y_train'], data['y_test'],
                   data.get('feature_names'), data.get('class_names'))
        
        elif isinstance(data, pd.DataFrame):
            # Данные представлены в виде DataFrame
            print("Обнаружен DataFrame")
            y_col = 'loyalty_class'
            if y_col not in data.columns:
                raise ValueError(f"Колонка целевой переменной '{y_col}' не найдена в данных")
            
            X = data.drop(y_col, axis=1)
            y = data[y_col]
            feature_names = X.columns.tolist()
            class_names = sorted(y.unique())
            
            return X, y, feature_names, class_names
        
        else:
            # Непонятный формат данных
            raise ValueError("Неизвестный формат данных в файле pkl")
    
    elif data_path.endswith('.csv'):
        # Загрузка из CSV
        data = pd.read_csv(data_path)
        y_col = 'loyalty_class'
        
        if y_col not in data.columns:
            raise ValueError(f"Колонка целевой переменной '{y_col}' не найдена в данных")
        
        X = data.drop(y_col, axis=1)
        y = data[y_col]
        feature_names = X.columns.tolist()
        class_names = sorted(y.unique())
        
        return X, y, feature_names, class_names
    
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {data_path}")


def prepare_data(X, y, test_size=0.2, random_state=42, scale=True):
    """
    Подготовка данных для обучения моделей.
    
    Args:
        X: Признаки.
        y: Целевая переменная.
        test_size (float): Размер тестовой выборки.
        random_state (int): Seed для генератора случайных чисел.
        scale (bool): Применять ли стандартизацию признаков.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("Подготовка данных...")
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Стандартизация признаков (если требуется)
    if scale:
        print("Применение стандартизации признаков...")
        scaler = StandardScaler()
        
        if isinstance(X_train, pd.DataFrame):
            # Для DataFrame
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            # Для numpy arrays
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names=None, class_names=None, 
                             balance_classes=False, use_ensemble=False, tune_models=False,
                             n_top_features=20, output_dir='../output/demo'):
    """
    Обучение и оценка моделей классификации.
    
    Args:
        X_train: Обучающая выборка (признаки).
        X_test: Тестовая выборка (признаки).
        y_train: Обучающая выборка (целевая переменная).
        y_test: Тестовая выборка (целевая переменная).
        feature_names (list): Имена признаков.
        class_names (list): Имена классов.
        balance_classes (bool): Использовать веса классов для балансировки.
        use_ensemble (bool): Создать ансамблевые модели.
        tune_models (bool): Настроить гиперпараметры моделей.
        n_top_features (int): Количество топ-признаков для визуализации.
        output_dir (str): Директория для сохранения результатов.
        
    Returns:
        tuple: trainer, evaluator - объекты для обучения и оценки моделей.
    """
    # Создание директорий для результатов
    models_dir = os.path.join(output_dir, 'models')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Вывод основной информации о данных
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Количество классов: {len(np.unique(y_train))}")
    print(f"Распределение классов (обучающая выборка):")
    for cls, count in zip(*np.unique(y_train, return_counts=True)):
        print(f"  Класс {cls}: {count} ({count/len(y_train)*100:.2f}%)")
    
    # Создание тренера моделей
    trainer = ModelTrainer(models_dir=models_dir, results_dir=results_dir)
    
    # Определение весов классов для балансировки (если требуется)
    class_weights = None
    if balance_classes:
        print("Расчет весов классов для балансировки...")
        class_counts = np.bincount(y_train)
        n_classes = len(class_counts)
        class_weights = {i: len(y_train) / (n_classes * count) for i, count in enumerate(class_counts)}
        print(f"Веса классов: {class_weights}")
    
    # Добавление базовых моделей
    print("\nДобавление базовых моделей...")
    trainer.add_base_models(class_weights=class_weights)
    
    # Обучение моделей
    print("\nОбучение моделей...")
    trainer.train_all_models(X_train, y_train)
    
    # Настройка гиперпараметров (если требуется)
    if tune_models:
        print("\nНастройка гиперпараметров...")
        param_grids = get_hyperparameter_grids()
        
        # Для демо используем более легкую настройку
        for name, param_grid in param_grids.items():
            if name in trainer.models:
                # Упрощаем сетки для демонстрации
                simplified_grid = {}
                for param, values in param_grid.items():
                    if isinstance(values, list) and len(values) > 3:
                        # Берем только часть значений
                        simplified_grid[param] = values[:3]
                    else:
                        simplified_grid[param] = values
                
                print(f"Настройка гиперпараметров для {name}...")
                try:
                    trainer.tune_hyperparameters(
                        name, X_train, y_train, simplified_grid, 
                        method='random', n_iter=5, cv=3
                    )
                    best_params = trainer.results[name]['tuning']['best_params']
                    print(f"  Лучшие параметры: {best_params}")
                except Exception as e:
                    print(f"  Ошибка при настройке {name}: {e}")
    
    # Создание ансамблевых моделей (если требуется)
    if use_ensemble:
        print("\nСоздание ансамблевых моделей...")
        
        # Отбор базовых моделей для ансамбля
        base_models = [(name, model) for name, model in trainer.models.items() 
                       if name not in ['voting_ensemble', 'stacking_ensemble']]
        
        # Создание ансамбля голосования
        print("  Создание ансамбля голосования...")
        trainer.create_voting_ensemble(estimators=base_models, voting='soft')
        trainer.train_model('voting_ensemble', X_train, y_train)
        
        # Создание ансамбля стекинга
        print("  Создание ансамбля стекинга...")
        trainer.create_stacking_ensemble(estimators=base_models)
        trainer.train_model('stacking_ensemble', X_train, y_train)
    
    # Оценка моделей
    print("\nОценка моделей на тестовой выборке...")
    results = trainer.evaluate_all_models(X_test, y_test)
    
    # Сравнение моделей
    comparison = trainer.compare_models(metric='f1_macro')
    print("\nСравнение моделей:")
    print(comparison.to_string())
    
    # Создание оценщика моделей для визуализации
    print("\nВизуализация результатов...")
    evaluator = ModelEvaluator(trainer.models, results_dir=results_dir)
    
    # Оценка моделей с расширенными метриками
    evaluator.evaluate_models(X_test, y_test, class_names)
    
    # Построение матриц ошибок
    print("  Построение матриц ошибок...")
    cm_fig = evaluator.plot_confusion_matrices(X_test, y_test, class_names)
    
    # Построение ROC-кривых
    print("  Построение ROC-кривых...")
    roc_fig = evaluator.plot_roc_curves(X_test, y_test)
    
    # Построение PR-кривых
    print("  Построение PR-кривых...")
    pr_fig = evaluator.plot_precision_recall_curves(X_test, y_test)
    
    # Важность признаков
    print(f"  Визуализация важности признаков (топ-{n_top_features})...")
    fi_fig = evaluator.plot_feature_importance(X_train, feature_names, top_n=n_top_features)
    
    # Сравнение метрик
    print("  Сравнение метрик качества...")
    metrics_fig = evaluator.plot_metrics_comparison()
    
    # Распределение классов
    print("  Распределение классов...")
    y_pred_dict = {name: model.predict(X_test) for name, model in trainer.models.items()}
    class_fig = evaluator.plot_class_distribution(y_test, y_pred_dict)
    
    # Генерация отчета
    print("\nГенерация отчета...")
    report = evaluator.generate_evaluation_report()
    
    # Сохранение результатов
    print("\nСохранение результатов...")
    evaluator.save_evaluation_results()
    model_paths = trainer.save_all_models()
    
    # Вывод лучшей модели
    best_model, best_score = evaluator.get_best_model(metric='f1_macro')
    print(f"\nЛучшая модель по F1-мере: {best_model} (F1 = {best_score:.4f})")
    
    return trainer, evaluator


def main():
    """
    Основная функция демонстрационного скрипта.
    """
    # Парсинг аргументов
    args = parse_args()
    
    # Создание директории для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Загрузка данных
        data = load_dataset(args.data_path)
        
        if len(data) == 6:
            # Данные уже разделены
            X_train, X_test, y_train, y_test, feature_names, class_names = data
        else:
            # Данные нужно подготовить
            X, y, feature_names, class_names = data
            X_train, X_test, y_train, y_test = prepare_data(
                X, y, test_size=args.test_size, random_state=args.random_state
            )
        
        # Обучение и оценка моделей
        trainer, evaluator = train_and_evaluate_models(
            X_train, X_test, y_train, y_test,
            feature_names=feature_names,
            class_names=class_names,
            balance_classes=args.balance_classes,
            use_ensemble=args.use_ensemble,
            tune_models=args.tune_models,
            n_top_features=args.n_top_features,
            output_dir=args.output_dir
        )
        
        print(f"\nДемонстрация завершена. Результаты сохранены в {args.output_dir}")
        return 0
    
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 