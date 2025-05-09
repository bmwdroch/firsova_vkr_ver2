#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для расширенной настройки гиперпараметров моделей классификации лояльности клиентов.
Включает различные стратегии оптимизации: Bayesian, Random Search, Grid Search и модифицированные поисковые пространства.
"""

import numpy as np
import pandas as pd
import time
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import joblib

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not found. Bayesian optimization will not be available.")
    print("Install it using: pip install scikit-optimize")

class HyperparameterTuner:
    """
    Класс для расширенной настройки гиперпараметров с использованием различных стратегий оптимизации.
    Поддерживает Grid Search, Random Search и Bayesian Optimization (если доступно).
    """
    
    def __init__(self, output_dir='../../output/models', results_dir='../../output/results/tuning'):
        """
        Инициализация настройщика гиперпараметров.
        
        Args:
            output_dir (str): Директория для сохранения моделей.
            results_dir (str): Директория для сохранения результатов настройки.
        """
        self.output_dir = output_dir
        self.results_dir = results_dir
        self.results = {}
        
        # Создаем директории, если они не существуют
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    def get_param_grid(self, model_name, search_type='random', n_iterations=None):
        """
        Возвращает оптимальное пространство поиска для заданной модели и типа поиска.
        
        Args:
            model_name (str): Имя модели ('logistic_regression', 'xgboost', 'lightgbm', etc.)
            search_type (str): Тип поиска ('grid', 'random', 'bayesian')
            n_iterations (int): Число итераций для Random/Bayesian поиска, 
                               влияет на детализацию сетки
                               
        Returns:
            dict or list: Пространство поиска гиперпараметров
        """
        if search_type == 'bayesian' and not SKOPT_AVAILABLE:
            print("Warning: Bayesian optimization not available. Falling back to random search.")
            search_type = 'random'
        
        # Настраиваем детализацию пространства поиска в зависимости от типа поиска и числа итераций
        detail_level = 'low'
        if n_iterations:
            if n_iterations <= 10:
                detail_level = 'low'
            elif n_iterations <= 50:
                detail_level = 'medium'
            else:
                detail_level = 'high'
        
        # Пространства поиска для различных моделей
        param_spaces = {
            'logistic_regression': {
                'grid': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', None],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
                    'class_weight': [None, 'balanced']
                },
                'random': {
                    'C': np.logspace(-4, 4, 20),
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'class_weight': [None, 'balanced']
                },
                'bayesian': [
                    Real(1e-4, 1e4, prior='log-uniform', name='C'),
                    Categorical(['l1', 'l2', None], name='penalty'),
                    Categorical(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], name='solver'),
                    Categorical([None, 'balanced'], name='class_weight')
                ]
            },
            'random_forest': {
                'grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': [None, 'balanced', 'balanced_subsample']
                },
                'random': {
                    'n_estimators': np.arange(50, 500, 25),
                    'max_depth': [None] + list(np.arange(5, 30, 2)),
                    'min_samples_split': np.arange(2, 20, 2),
                    'min_samples_leaf': np.arange(1, 10, 1),
                    'max_features': ['sqrt', 'log2', None, 0.5, 0.7, 0.8],
                    'class_weight': [None, 'balanced', 'balanced_subsample']
                },
                'bayesian': [
                    Integer(50, 500, name='n_estimators'),
                    Integer(3, 30, name='max_depth'),
                    Integer(2, 20, name='min_samples_split'),
                    Integer(1, 10, name='min_samples_leaf'),
                    Categorical(['sqrt', 'log2', None, 0.5, 0.7, 0.8], name='max_features'),
                    Categorical([None, 'balanced', 'balanced_subsample'], name='class_weight')
                ]
            },
            'xgboost': {
                'grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'gamma': [0, 0.1, 0.2]
                },
                'random': {
                    'n_estimators': np.arange(50, 400, 25),
                    'learning_rate': np.logspace(-3, 0, 10),
                    'max_depth': np.arange(3, 12, 1),
                    'min_child_weight': np.arange(1, 10, 1),
                    'subsample': np.arange(0.5, 1.0, 0.05),
                    'colsample_bytree': np.arange(0.5, 1.0, 0.05),
                    'gamma': np.arange(0, 0.5, 0.05)
                },
                'bayesian': [
                    Integer(50, 400, name='n_estimators'),
                    Real(0.001, 0.3, prior='log-uniform', name='learning_rate'),
                    Integer(3, 12, name='max_depth'),
                    Integer(1, 10, name='min_child_weight'),
                    Real(0.5, 1.0, name='subsample'),
                    Real(0.5, 1.0, name='colsample_bytree'),
                    Real(0, 0.5, name='gamma')
                ]
            },
            'lightgbm': {
                'grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7, -1],
                    'num_leaves': [31, 63, 127],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'min_child_samples': [5, 10, 20]
                },
                'random': {
                    'n_estimators': np.arange(50, 500, 25),
                    'learning_rate': np.logspace(-3, 0, 10),
                    'max_depth': np.arange(3, 12, 1).tolist() + [-1],
                    'num_leaves': np.arange(15, 255, 10),
                    'subsample': np.arange(0.5, 1.0, 0.05),
                    'colsample_bytree': np.arange(0.5, 1.0, 0.05),
                    'min_child_samples': np.arange(5, 100, 5)
                },
                'bayesian': [
                    Integer(50, 500, name='n_estimators'),
                    Real(0.001, 0.3, prior='log-uniform', name='learning_rate'),
                    Categorical([-1] + list(range(3, 12)), name='max_depth'),
                    Integer(15, 255, name='num_leaves'),
                    Real(0.5, 1.0, name='subsample'),
                    Real(0.5, 1.0, name='colsample_bytree'),
                    Integer(5, 100, name='min_child_samples')
                ]
            },
            'svm': {
                'grid': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.1],
                    'kernel': ['linear', 'rbf'],
                    'class_weight': [None, 'balanced']
                },
                'random': {
                    'C': np.logspace(-2, 3, 10),
                    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 10)),
                    'kernel': ['linear', 'rbf', 'poly'],
                    'class_weight': [None, 'balanced']
                },
                'bayesian': [
                    Real(0.01, 1000, prior='log-uniform', name='C'),
                    Categorical(['scale', 'auto'] + list(np.logspace(-3, 2, 5)), name='gamma'),
                    Categorical(['linear', 'rbf', 'poly'], name='kernel'),
                    Categorical([None, 'balanced'], name='class_weight')
                ]
            }
        }
        
        # Если модель не найдена в словаре, возвращаем пустой словарь
        if model_name not in param_spaces:
            print(f"Warning: No predefined parameter grid for model '{model_name}'.")
            return {}
        
        # Возвращаем пространство поиска для заданного типа поиска
        return param_spaces[model_name][search_type]
        
    def tune_hyperparameters(self, model, model_name, X_train, y_train, method='random', 
                           param_grid=None, cv=5, n_iter=20, scoring='f1_macro', 
                           n_jobs=-1, verbose=1, random_state=42):
        """
        Настройка гиперпараметров модели с использованием различных стратегий оптимизации.
        
        Args:
            model: Модель для настройки
            model_name (str): Имя модели для сохранения результатов
            X_train: Обучающие данные
            y_train: Целевые значения
            method (str): Метод поиска ('grid', 'random', 'bayesian')
            param_grid (dict): Сетка параметров (если None, используется предопределенная)
            cv (int or cv generator): Кросс-валидация
            n_iter (int): Количество итераций для random и bayesian поиска
            scoring (str): Метрика для оптимизации
            n_jobs (int): Количество используемых процессов
            verbose (int): Уровень детализации вывода
            random_state (int): Случайное зерно
            
        Returns:
            Model: Модель с оптимальными гиперпараметрами
        """
        # Проверка метода
        if method not in ['grid', 'random', 'bayesian']:
            print(f"Warning: Unknown method '{method}'. Falling back to 'random'.")
            method = 'random'
        
        if method == 'bayesian' and not SKOPT_AVAILABLE:
            print("Warning: Bayesian optimization not available. Falling back to random search.")
            method = 'random'
        
        # Если сетка параметров не задана, используем предопределенную
        if param_grid is None:
            param_grid = self.get_param_grid(model_name, method, n_iter)
        
        # Настройка стратифицированной кросс-валидации (если cv - число)
        if isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        start_time = time.time()
        
        # Выполнение поиска оптимальных параметров
        if method == 'grid':
            search = GridSearchCV(
                model, param_grid, scoring=scoring, cv=cv, 
                n_jobs=n_jobs, verbose=verbose, return_train_score=True
            )
            search.fit(X_train, y_train)
            
        elif method == 'random':
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, scoring=scoring, cv=cv,
                n_jobs=n_jobs, verbose=verbose, random_state=random_state, 
                return_train_score=True
            )
            search.fit(X_train, y_train)
            
        elif method == 'bayesian':
            # Для Bayesian optimization нужен специальный формат пространства поиска
            search = BayesSearchCV(
                model, param_grid, n_iter=n_iter, scoring=scoring, cv=cv,
                n_jobs=n_jobs, verbose=verbose, random_state=random_state,
                return_train_score=True
            )
            search.fit(X_train, y_train)
        
        tuning_time = time.time() - start_time
        
        # Сохранение результатов настройки
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'tuning_time': tuning_time,
            'method': method,
            'cv_results': {
                'params': search.cv_results_['params'],
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist(),
                'mean_train_score': search.cv_results_['mean_train_score'].tolist(),
                'std_train_score': search.cv_results_['std_train_score'].tolist(),
                'rank_test_score': search.cv_results_['rank_test_score'].tolist()
            }
        }
        
        # Дополнительные данные для bayesian, если доступно
        if method == 'bayesian' and hasattr(search, 'optimizer_results_'):
            results['bayesian_results'] = {
                'x_iters': [list(map(str, x)) for x in search.optimizer_results_.x_iters],
                'func_vals': search.optimizer_results_.func_vals.tolist()
            }
        
        # Сохраняем результаты в JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(self.results_dir, f'{model_name}_{method}_{timestamp}.json')
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Сохраняем модель с лучшими параметрами
        model_path = os.path.join(self.output_dir, f'{model_name}_tuned_{timestamp}.pkl')
        joblib.dump(search.best_estimator_, model_path)
        
        # Запоминаем результаты
        self.results[model_name] = results
        
        # Визуализируем результаты настройки
        self.visualize_tuning_results(model_name, results, method)
        
        return search.best_estimator_
    
    def visualize_tuning_results(self, model_name, results, method):
        """
        Визуализирует результаты настройки гиперпараметров.
        
        Args:
            model_name (str): Имя модели
            results (dict): Результаты настройки
            method (str): Метод поиска
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        vis_dir = os.path.join(self.results_dir, 'plots')
        
        # Построение графика распределения оценок для разных комбинаций параметров
        plt.figure(figsize=(12, 6))
        
        # Сортировка по убыванию средней оценки
        indices = np.argsort(results['cv_results']['mean_test_score'])[::-1]
        sorted_scores = np.array(results['cv_results']['mean_test_score'])[indices]
        sorted_stds = np.array(results['cv_results']['std_test_score'])[indices]
        
        # Ограничиваем количество отображаемых конфигураций
        num_configs = min(20, len(sorted_scores))
        
        plt.errorbar(range(num_configs), sorted_scores[:num_configs], 
                     yerr=sorted_stds[:num_configs], fmt='o-', capsize=5)
        plt.title(f'Top {num_configs} Parameter Configurations for {model_name} ({method})')
        plt.xlabel('Configuration Rank')
        plt.ylabel(f'Mean CV Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(vis_dir, f'{model_name}_{method}_scores_{timestamp}.png'))
        plt.close()
        
        # Для bayesian метода добавляем график сходимости
        if method == 'bayesian' and 'bayesian_results' in results:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(results['bayesian_results']['func_vals'])+1), 
                     -np.array(results['bayesian_results']['func_vals']))
            plt.title(f'Bayesian Optimization Convergence for {model_name}')
            plt.xlabel('Iterations')
            plt.ylabel('Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(os.path.join(vis_dir, f'{model_name}_bayesian_convergence_{timestamp}.png'))
            plt.close()
        
        # Анализ важности параметров (для случайного поиска)
        if method in ['random', 'bayesian'] and len(results['cv_results']['params']) > 5:
            # Подготовка данных для анализа
            param_data = []
            for param_set, score in zip(results['cv_results']['params'], 
                                       results['cv_results']['mean_test_score']):
                row = {'score': score}
                for param, value in param_set.items():
                    # Преобразуем значения в строки для категориальных признаков
                    row[param] = str(value) if not isinstance(value, (int, float)) else value
                param_data.append(row)
            
            # Создаем DataFrame
            param_df = pd.DataFrame(param_data)
            
            # Выявляем числовые и категориальные параметры
            numeric_params = [col for col in param_df.columns if col != 'score' 
                             and pd.api.types.is_numeric_dtype(param_df[col])]
            
            categorical_params = [col for col in param_df.columns if col != 'score' 
                                 and col not in numeric_params]
            
            # Для числовых параметров делаем scatter plot
            if numeric_params:
                fig, axes = plt.subplots(1, len(numeric_params), figsize=(5*len(numeric_params), 5))
                if len(numeric_params) == 1:
                    axes = [axes]
                
                for i, param in enumerate(numeric_params):
                    axes[i].scatter(param_df[param], param_df['score'], alpha=0.7)
                    axes[i].set_xlabel(param)
                    axes[i].set_ylabel('Score')
                    axes[i].set_title(f'Score vs {param}')
                    axes[i].grid(True, linestyle='--', alpha=0.7)
                
                fig.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'{model_name}_{method}_numeric_params_{timestamp}.png'))
                plt.close()
            
            # Для категориальных параметров делаем box plot
            if categorical_params:
                fig, axes = plt.subplots(1, len(categorical_params), figsize=(5*len(categorical_params), 5))
                if len(categorical_params) == 1:
                    axes = [axes]
                
                for i, param in enumerate(categorical_params):
                    sns.boxplot(x=param, y='score', data=param_df, ax=axes[i])
                    axes[i].set_xlabel(param)
                    axes[i].set_ylabel('Score')
                    axes[i].set_title(f'Score vs {param}')
                    if len(param_df[param].unique()) > 5:
                        axes[i].tick_params(axis='x', rotation=45)
                
                fig.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'{model_name}_{method}_categorical_params_{timestamp}.png'))
                plt.close()
    
    def tune_multiple_models(self, models, X_train, y_train, method='random', cv=5, 
                            n_iter=20, scoring='f1_macro', n_jobs=-1, verbose=1):
        """
        Настройка нескольких моделей и сравнение их результатов.
        
        Args:
            models (dict): Словарь {имя_модели: модель}
            X_train: Обучающие данные
            y_train: Целевые значения
            method (str): Метод поиска
            cv (int): Число фолдов для кросс-валидации
            n_iter (int): Число итераций для random и bayesian поиска
            scoring (str): Метрика для оптимизации
            n_jobs (int): Число используемых процессов
            verbose (int): Уровень детализации вывода
            
        Returns:
            dict: Словарь {имя_модели: настроенная_модель}
        """
        tuned_models = {}
        
        for model_name, model in models.items():
            print(f"Настройка модели: {model_name}")
            try:
                tuned_model = self.tune_hyperparameters(
                    model, model_name, X_train, y_train, 
                    method=method, cv=cv, n_iter=n_iter, 
                    scoring=scoring, n_jobs=n_jobs, verbose=verbose
                )
                tuned_models[model_name] = tuned_model
                print(f"  Лучшие параметры: {self.results[model_name]['best_params']}")
                print(f"  Лучший счет: {self.results[model_name]['best_score']:.4f}")
                print(f"  Время настройки: {self.results[model_name]['tuning_time']:.2f} сек")
            except Exception as e:
                print(f"  Ошибка при настройке {model_name}: {e}")
        
        # Сравнение настроенных моделей
        self.compare_tuned_models()
        
        return tuned_models
    
    def compare_tuned_models(self):
        """
        Сравнивает результаты настройки различных моделей.
        
        Returns:
            pandas.DataFrame: Сравнительная таблица моделей
        """
        if not self.results:
            print("Нет результатов для сравнения.")
            return None
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {
                'model': model_name,
                'best_score': results['best_score'],
                'tuning_time': results['tuning_time'],
                'method': results['method']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values(by='best_score', ascending=False)
        
        # Визуализация сравнения
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y='best_score', data=df, hue='method')
        plt.title('Comparison of Tuned Models')
        plt.xlabel('Model')
        plt.ylabel('Best Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Сохранение визуализации
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        vis_dir = os.path.join(self.results_dir, 'plots')
        plt.savefig(os.path.join(vis_dir, f'model_comparison_{timestamp}.png'))
        plt.close()
        
        # Вывод сравнительной таблицы
        print("\nСравнение настроенных моделей:")
        print(df.to_string(index=False))
        
        return df

# Функция для получения пространства параметров для различных моделей
def get_hyperparameter_spaces(model_name, search_type='random', detail_level='medium'):
    """
    Возвращает пространства поиска гиперпараметров для заданной модели и типа поиска.
    
    Args:
        model_name (str): Имя модели
        search_type (str): Тип поиска ('grid', 'random', 'bayesian')
        detail_level (str): Уровень детализации ('low', 'medium', 'high')
        
    Returns:
        dict or list: Пространство поиска гиперпараметров
    """
    tuner = HyperparameterTuner()
    n_iterations = None
    
    if detail_level == 'low':
        n_iterations = 10
    elif detail_level == 'medium':
        n_iterations = 30
    else:  # high
        n_iterations = 100
    
    return tuner.get_param_grid(model_name, search_type, n_iterations) 