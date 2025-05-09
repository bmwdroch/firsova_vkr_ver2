import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from datetime import datetime
import json

# Импорт собственных модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.model_training import ModelTrainer, get_hyperparameter_grids
from modeling.model_evaluation import ModelEvaluator


def parse_args():
    """
    Парсинг аргументов командной строки.
    
    Returns:
        argparse.Namespace: Аргументы командной строки.
    """
    parser = argparse.ArgumentParser(description='Обучение и оценка моделей классификации лояльности клиентов')
    
    parser.add_argument('--data_path', type=str, default='../output/loyalty_dataset.pkl',
                       help='Путь к подготовленному датасету')
    parser.add_argument('--models_dir', type=str, default='../output/models',
                       help='Директория для сохранения моделей')
    parser.add_argument('--results_dir', type=str, default='../output/results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Размер тестовой выборки (если данные еще не разделены)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Seed для генератора случайных чисел')
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='Выполнить настройку гиперпараметров моделей')
    parser.add_argument('--tune_method', type=str, default='random', choices=['grid', 'random'],
                       help='Метод настройки гиперпараметров')
    parser.add_argument('--n_iter', type=int, default=20,
                       help='Количество итераций для RandomizedSearchCV')
    parser.add_argument('--cv', type=int, default=5,
                       help='Количество фолдов для кросс-валидации')
    parser.add_argument('--create_ensemble', action='store_true',
                       help='Создать ансамблевые модели')
    parser.add_argument('--ensemble_type', type=str, default='voting', choices=['voting', 'stacking', 'both'],
                       help='Тип ансамблевой модели')
    
    return parser.parse_args()


def load_data(data_path):
    """
    Загрузка подготовленного датасета.
    
    Args:
        data_path (str): Путь к файлу с данными.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names, class_names.
    """
    print(f"Загрузка данных из {data_path}...")
    
    if data_path.endswith('.pkl'):
        # Загрузка из пикл-файла
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            # Предполагаем, что данные уже разделены на выборки
            X_train = data.get('X_train')
            X_test = data.get('X_test')
            y_train = data.get('y_train')
            y_test = data.get('y_test')
            feature_names = data.get('feature_names')
            class_names = data.get('class_names')
            
            return X_train, X_test, y_train, y_test, feature_names, class_names
        else:
            # Данные представлены в виде DataFrame
            X = data.drop('loyalty_class', axis=1, errors='ignore')
            y = data['loyalty_class'] if 'loyalty_class' in data.columns else None
            feature_names = X.columns.tolist()
            class_names = np.unique(y) if y is not None else None
            
            return X, None, y, None, feature_names, class_names
    
    elif data_path.endswith('.csv'):
        # Загрузка из CSV
        data = pd.read_csv(data_path)
        X = data.drop('loyalty_class', axis=1, errors='ignore')
        y = data['loyalty_class'] if 'loyalty_class' in data.columns else None
        feature_names = X.columns.tolist()
        class_names = np.unique(y) if y is not None else None
        
        return X, None, y, None, feature_names, class_names
    
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {data_path}")


def save_config(args, config_dir):
    """
    Сохраняет конфигурацию запуска в JSON-файл.
    
    Args:
        args (argparse.Namespace): Аргументы командной строки.
        config_dir (str): Директория для сохранения.
        
    Returns:
        str: Путь к сохраненному файлу.
    """
    os.makedirs(config_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(config_dir, f'training_config_{timestamp}.json')
    
    config = vars(args)
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    
    return filename


def main():
    """
    Основная функция для обучения и оценки моделей.
    """
    # Парсинг аргументов
    args = parse_args()
    
    # Создание директорий для результатов
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Сохранение конфигурации
    config_path = save_config(args, args.results_dir)
    print(f"Конфигурация сохранена в {config_path}")
    
    # Загрузка данных
    data = load_data(args.data_path)
    
    if len(data) == 6:
        X_train, X_test, y_train, y_test, feature_names, class_names = data
    else:
        X, _, y, _, feature_names, class_names = data
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Классы: {class_names}")
    
    # Создание тренера моделей
    trainer = ModelTrainer(models_dir=args.models_dir, results_dir=args.results_dir)
    
    # Определение весов классов для учета несбалансированности
    class_counts = np.bincount(y_train)
    class_weights = {i: len(y_train) / (len(class_names) * count) for i, count in enumerate(class_counts)}
    
    # Добавление базовых моделей
    print("Добавление базовых моделей...")
    trainer.add_base_models(class_weights=class_weights)
    
    # Обучение всех моделей
    print("Обучение моделей...")
    trainer.train_all_models(X_train, y_train)
    
    # Настройка гиперпараметров (если требуется)
    if args.tune_hyperparams:
        print("Настройка гиперпараметров...")
        param_grids = get_hyperparameter_grids()
        
        for name, param_grid in param_grids.items():
            if name in trainer.models:
                print(f"Настройка гиперпараметров для {name}...")
                try:
                    trainer.tune_hyperparameters(
                        name, X_train, y_train, param_grid, 
                        cv=args.cv, method=args.tune_method, n_iter=args.n_iter
                    )
                except Exception as e:
                    print(f"Ошибка при настройке гиперпараметров для {name}: {e}")
    
    # Создание ансамблевых моделей (если требуется)
    if args.create_ensemble:
        print("Создание ансамблевых моделей...")
        
        # Отбор лучших моделей для ансамбля
        models_for_ensemble = []
        for name, model in trainer.models.items():
            if name not in ['voting_ensemble', 'stacking_ensemble']:
                models_for_ensemble.append((name, model))
        
        if args.ensemble_type == 'voting' or args.ensemble_type == 'both':
            print("Создание ансамбля голосования...")
            trainer.create_voting_ensemble(estimators=models_for_ensemble, voting='soft')
            trainer.train_model('voting_ensemble', X_train, y_train)
        
        if args.ensemble_type == 'stacking' or args.ensemble_type == 'both':
            print("Создание ансамбля стекинга...")
            trainer.create_stacking_ensemble(estimators=models_for_ensemble)
            trainer.train_model('stacking_ensemble', X_train, y_train)
    
    # Оценка моделей
    print("Оценка моделей на тестовой выборке...")
    results = trainer.evaluate_all_models(X_test, y_test)
    
    # Сравнение моделей
    comparison = trainer.compare_models(metric='f1_macro')
    print("\nСравнение моделей:")
    print(comparison)
    
    # Сохранение моделей
    print("\nСохранение моделей...")
    model_paths = trainer.save_all_models()
    
    # Сохранение результатов
    print("Сохранение результатов оценки...")
    results_path = trainer.save_results()
    
    # Визуализация результатов
    print("Визуализация результатов...")
    evaluator = ModelEvaluator(trainer.models, results_dir=args.results_dir)
    
    # Оценка моделей с дополнительными метриками
    evaluator.evaluate_models(X_test, y_test, class_names)
    
    # Построение матриц ошибок
    confusion_matrices_fig = evaluator.plot_confusion_matrices(X_test, y_test, class_names)
    
    # Построение ROC-кривых
    roc_curves_fig = evaluator.plot_roc_curves(X_test, y_test)
    
    # Построение PR-кривых
    pr_curves_fig = evaluator.plot_precision_recall_curves(X_test, y_test)
    
    # Построение важности признаков
    feature_importance_fig = evaluator.plot_feature_importance(X_train, feature_names, top_n=20)
    
    # Построение сравнения метрик
    metrics_comparison_fig = evaluator.plot_metrics_comparison()
    
    # Распределение классов
    y_pred_dict = {name: model.predict(X_test) for name, model in trainer.models.items()}
    class_distribution_fig = evaluator.plot_class_distribution(y_test, y_pred_dict)
    
    # Генерация отчета
    print("Генерация итогового отчета...")
    report = evaluator.generate_evaluation_report()
    
    # Сохранение результатов оценки
    evaluator.save_evaluation_results()
    
    print(f"\nОбучение и оценка моделей завершены. Результаты сохранены в {args.results_dir}")
    
    # Отображение лучшей модели
    best_model, best_score = evaluator.get_best_model(metric='f1_macro')
    print(f"Лучшая модель по F1-мере: {best_model} (F1 = {best_score:.4f})")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 