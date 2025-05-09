#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для расширенной валидации моделей классификации лояльности клиентов.
Включает различные методы кросс-валидации и анализа стабильности моделей.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_validate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
import joblib

class ModelValidator:
    """
    Класс для расширенной валидации моделей с применением различных стратегий 
    кросс-валидации и оценки стабильности моделей на разных выборках.
    """
    
    def __init__(self, output_dir='../../output/models', results_dir='../../output/results/validation'):
        """
        Инициализация валидатора моделей.
        
        Args:
            output_dir (str): Директория для сохранения моделей.
            results_dir (str): Директория для сохранения результатов валидации.
        """
        self.output_dir = output_dir
        self.results_dir = results_dir
        self.results = {}
        
        # Создаем директории, если они не существуют
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    def stratified_cross_validation(self, model, model_name, X, y, cv=5, scoring=None, 
                                  n_jobs=-1, return_estimator=False):
        """
        Проводит стратифицированную кросс-валидацию модели.
        
        Args:
            model: Модель для валидации
            model_name (str): Имя модели для сохранения результатов
            X: Признаки
            y: Целевая переменная
            cv (int): Количество фолдов
            scoring (str, list): Метрики для оценки
            n_jobs (int): Количество используемых процессов
            return_estimator (bool): Возвращать обученные модели для каждого фолда
            
        Returns:
            dict: Результаты кросс-валидации
        """
        if scoring is None:
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Создаем генератор стратифицированных фолдов
        stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Проводим кросс-валидацию
        cv_results = cross_validate(
            model, X, y, cv=stratified_cv, scoring=scoring, 
            n_jobs=n_jobs, return_train_score=True,
            return_estimator=return_estimator
        )
        
        # Обработка и сохранение результатов
        validation_results = self._process_cv_results(cv_results, scoring, model_name, "stratified")
        
        # Визуализация результатов
        self._visualize_cv_results(validation_results, model_name, "stratified")
        
        # Сохранение результатов в JSON
        self._save_validation_results(validation_results, model_name, "stratified")
        
        return validation_results
    
    def time_series_cross_validation(self, model, model_name, X, y, time_column=None, 
                                   n_splits=5, test_size=0.2, scoring=None, 
                                   n_jobs=-1, return_estimator=False):
        """
        Проводит временную кросс-валидацию модели.
        
        Args:
            model: Модель для валидации
            model_name (str): Имя модели для сохранения результатов
            X: Признаки
            y: Целевая переменная
            time_column (str, array): Временная колонка или массив для сортировки
            n_splits (int): Количество разбиений
            test_size (float): Размер тестовой выборки
            scoring (str, list): Метрики для оценки
            n_jobs (int): Количество используемых процессов
            return_estimator (bool): Возвращать обученные модели для каждого фолда
            
        Returns:
            dict: Результаты кросс-валидации
        """
        if scoring is None:
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Сортировка данных по времени, если указана временная колонка
        if time_column is not None:
            # Если time_column - имя колонки в X (pandas DataFrame)
            if isinstance(time_column, str) and isinstance(X, pd.DataFrame) and time_column in X.columns:
                sort_indices = np.argsort(X[time_column].values)
                X_sorted = X.iloc[sort_indices].copy()
                y_sorted = y[sort_indices] if isinstance(y, np.ndarray) else y.iloc[sort_indices]
            # Если time_column - отдельный массив
            elif isinstance(time_column, (list, np.ndarray)) and len(time_column) == len(X):
                sort_indices = np.argsort(time_column)
                X_sorted = X[sort_indices] if isinstance(X, np.ndarray) else X.iloc[sort_indices]
                y_sorted = y[sort_indices] if isinstance(y, np.ndarray) else y.iloc[sort_indices]
            else:
                print("Предупреждение: Недопустимый формат time_column. Используем исходный порядок данных.")
                X_sorted, y_sorted = X, y
        else:
            X_sorted, y_sorted = X, y
        
        # Создаем генератор временных фолдов
        time_cv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(X) * test_size))
        
        # Проводим кросс-валидацию
        cv_results = cross_validate(
            model, X_sorted, y_sorted, cv=time_cv, scoring=scoring, 
            n_jobs=n_jobs, return_train_score=True,
            return_estimator=return_estimator
        )
        
        # Обработка и сохранение результатов
        validation_results = self._process_cv_results(cv_results, scoring, model_name, "time_series")
        
        # Визуализация результатов
        self._visualize_cv_results(validation_results, model_name, "time_series")
        
        # Сохранение результатов в JSON
        self._save_validation_results(validation_results, model_name, "time_series")
        
        return validation_results
    
    def evaluate_model_stability(self, model, model_name, X, y, n_runs=10, 
                                test_size=0.2, stratify=True, scoring=None, random_seed=42):
        """
        Оценивает стабильность модели путем многократного обучения и тестирования 
        на разных случайных разбиениях данных.
        
        Args:
            model: Модель для оценки
            model_name (str): Имя модели для сохранения результатов
            X: Признаки
            y: Целевая переменная
            n_runs (int): Количество прогонов
            test_size (float): Размер тестовой выборки
            stratify (bool): Использовать стратификацию при разбиении
            scoring (list): Метрики для оценки
            random_seed (int): Базовое случайное зерно
            
        Returns:
            dict: Результаты оценки стабильности
        """
        from sklearn.model_selection import train_test_split
        
        if scoring is None:
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Инициализация массивов для сохранения результатов
        results = {metric: [] for metric in scoring}
        results['run_id'] = []
        
        # Выполнение n_runs прогонов
        for run in range(n_runs):
            # Установка случайного зерна для воспроизводимости
            run_seed = random_seed + run
            
            # Разбиение данных
            stratify_param = y if stratify else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=run_seed, stratify=stratify_param
            )
            
            # Обучение модели
            model_clone = clone_model(model)
            model_clone.fit(X_train, y_train)
            
            # Предсказание и оценка
            y_pred = model_clone.predict(X_test)
            
            # Расчет метрик
            results['run_id'].append(run + 1)
            for metric in scoring:
                if metric == 'accuracy':
                    score = accuracy_score(y_test, y_pred)
                elif metric == 'precision_macro':
                    score = precision_score(y_test, y_pred, average='macro')
                elif metric == 'recall_macro':
                    score = recall_score(y_test, y_pred, average='macro')
                elif metric == 'f1_macro':
                    score = f1_score(y_test, y_pred, average='macro')
                else:
                    score = 0  # Неизвестная метрика
                
                results[metric].append(score)
        
        # Преобразование результатов в DataFrame
        results_df = pd.DataFrame(results)
        
        # Расчет статистик стабильности
        stability_stats = {}
        for metric in scoring:
            stability_stats[f'{metric}_mean'] = results_df[metric].mean()
            stability_stats[f'{metric}_std'] = results_df[metric].std()
            stability_stats[f'{metric}_min'] = results_df[metric].min()
            stability_stats[f'{metric}_max'] = results_df[metric].max()
            stability_stats[f'{metric}_range'] = results_df[metric].max() - results_df[metric].min()
            stability_stats[f'{metric}_cv'] = results_df[metric].std() / results_df[metric].mean()  # Коэффициент вариации
        
        # Визуализация результатов
        self._visualize_stability_results(results_df, stability_stats, model_name)
        
        # Сохранение результатов
        stability_results = {
            'model': model_name,
            'n_runs': n_runs,
            'test_size': test_size,
            'stratify': stratify,
            'random_seed': random_seed,
            'run_metrics': results_df.to_dict(orient='records'),
            'stability_stats': stability_stats
        }
        
        # Сохранение в JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(self.results_dir, f'{model_name}_stability_{timestamp}.json')
        
        with open(results_path, 'w') as f:
            json.dump(stability_results, f, indent=4)
        
        self.results[f'{model_name}_stability'] = stability_results
        
        return stability_results
    
    def _process_cv_results(self, cv_results, scoring, model_name, method):
        """
        Обрабатывает результаты кросс-валидации.
        
        Args:
            cv_results (dict): Результаты кросс-валидации
            scoring (list): Метрики для оценки
            model_name (str): Имя модели
            method (str): Метод кросс-валидации
            
        Returns:
            dict: Обработанные результаты
        """
        processed_results = {
            'model': model_name,
            'method': method,
            'metrics': {}
        }
        
        # Обработка метрик
        for metric in scoring:
            train_scores = cv_results[f'train_{metric}']
            test_scores = cv_results[f'test_{metric}']
            
            processed_results['metrics'][metric] = {
                'train_mean': float(train_scores.mean()),
                'train_std': float(train_scores.std()),
                'test_mean': float(test_scores.mean()),
                'test_std': float(test_scores.std()),
                'train_values': train_scores.tolist(),
                'test_values': test_scores.tolist()
            }
        
        # Добавление времени выполнения
        if 'fit_time' in cv_results:
            processed_results['fit_time_mean'] = float(cv_results['fit_time'].mean())
            processed_results['fit_time_std'] = float(cv_results['fit_time'].std())
        
        if 'score_time' in cv_results:
            processed_results['score_time_mean'] = float(cv_results['score_time'].mean())
            processed_results['score_time_std'] = float(cv_results['score_time'].std())
        
        # Добавление информации о модели
        if 'estimator' in cv_results and len(cv_results['estimator']) > 0:
            estimator = cv_results['estimator'][0]
            if hasattr(estimator, 'get_params'):
                processed_results['model_params'] = estimator.get_params()
        
        return processed_results
    
    def _visualize_cv_results(self, validation_results, model_name, method):
        """
        Визуализирует результаты кросс-валидации.
        
        Args:
            validation_results (dict): Результаты валидации
            model_name (str): Имя модели
            method (str): Метод кросс-валидации
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        vis_dir = os.path.join(self.results_dir, 'plots')
        
        # Создание данных для визуализации
        metrics_data = []
        for metric, values in validation_results['metrics'].items():
            metrics_data.append({
                'metric': metric,
                'train_mean': values['train_mean'],
                'train_std': values['train_std'],
                'test_mean': values['test_mean'],
                'test_std': values['test_std']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Визуализация средних значений метрик с доверительными интервалами
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics_df))
        width = 0.35
        
        plt.bar(x - width/2, metrics_df['train_mean'], width, 
                label='Train', color='lightblue', 
                yerr=metrics_df['train_std'], capsize=5)
        
        plt.bar(x + width/2, metrics_df['test_mean'], width, 
                label='Test', color='salmon', 
                yerr=metrics_df['test_std'], capsize=5)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'{model_name} - {method.capitalize()} Cross-Validation Results')
        plt.xticks(x, metrics_df['metric'])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(vis_dir, f'{model_name}_{method}_cv_results_{timestamp}.png'))
        plt.close()
        
        # Визуализация распределения оценок для каждой метрики
        for metric, values in validation_results['metrics'].items():
            plt.figure(figsize=(8, 6))
            
            train_values = values['train_values']
            test_values = values['test_values']
            
            plt.boxplot([train_values, test_values], labels=['Train', 'Test'])
            plt.title(f'{model_name} - {metric} Distribution ({method.capitalize()} CV)')
            plt.ylabel('Score')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.savefig(os.path.join(vis_dir, f'{model_name}_{method}_{metric}_distribution_{timestamp}.png'))
            plt.close()
    
    def _visualize_stability_results(self, results_df, stability_stats, model_name):
        """
        Визуализирует результаты оценки стабильности модели.
        
        Args:
            results_df (DataFrame): Результаты по прогонам
            stability_stats (dict): Статистики стабильности
            model_name (str): Имя модели
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        vis_dir = os.path.join(self.results_dir, 'plots')
        
        # Визуализация метрик по прогонам
        metrics = [col for col in results_df.columns if col != 'run_id']
        
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            plt.plot(results_df['run_id'], results_df[metric], 'o-', label=metric)
        
        plt.xlabel('Run')
        plt.ylabel('Score')
        plt.title(f'{model_name} - Stability Across {len(results_df)} Runs')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(vis_dir, f'{model_name}_stability_runs_{timestamp}.png'))
        plt.close()
        
        # Визуализация распределения метрик
        plt.figure(figsize=(12, 6))
        
        # Преобразование данных для boxplot
        plot_data = []
        for metric in metrics:
            for value in results_df[metric]:
                plot_data.append({'Metric': metric, 'Value': value})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Построение boxplot
        sns.boxplot(x='Metric', y='Value', data=plot_df)
        plt.title(f'{model_name} - Distribution of Metrics Across Runs')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(vis_dir, f'{model_name}_stability_distribution_{timestamp}.png'))
        plt.close()
        
        # Визуализация коэффициентов вариации (CV)
        cv_data = {metric: stability_stats[f'{metric}_cv'] for metric in metrics}
        
        plt.figure(figsize=(10, 6))
        plt.bar(cv_data.keys(), cv_data.values(), color='lightblue')
        plt.xlabel('Metric')
        plt.ylabel('Coefficient of Variation')
        plt.title(f'{model_name} - Coefficient of Variation (Lower is More Stable)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(vis_dir, f'{model_name}_stability_cv_{timestamp}.png'))
        plt.close()
    
    def _save_validation_results(self, validation_results, model_name, method):
        """
        Сохраняет результаты валидации в JSON.
        
        Args:
            validation_results (dict): Результаты валидации
            model_name (str): Имя модели
            method (str): Метод валидации
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(self.results_dir, f'{model_name}_{method}_validation_{timestamp}.json')
        
        with open(results_path, 'w') as f:
            json.dump(validation_results, f, indent=4)
        
        self.results[f'{model_name}_{method}'] = validation_results
    
    def compare_validation_methods(self, model, model_name, X, y, time_column=None):
        """
        Сравнивает различные методы валидации для одной модели.
        
        Args:
            model: Модель для валидации
            model_name (str): Имя модели для сохранения результатов
            X: Признаки
            y: Целевая переменная
            time_column (str, array): Временная колонка или массив для сортировки
            
        Returns:
            dict: Результаты сравнения методов валидации
        """
        # Проводим различные типы валидации
        print(f"Выполнение стратифицированной кросс-валидации для {model_name}...")
        stratified_results = self.stratified_cross_validation(model, model_name, X, y)
        
        print(f"Выполнение временной кросс-валидации для {model_name}...")
        time_series_results = self.time_series_cross_validation(model, model_name, X, y, time_column)
        
        print(f"Оценка стабильности модели {model_name}...")
        stability_results = self.evaluate_model_stability(model, model_name, X, y)
        
        # Подготовка данных для сравнения
        comparison_data = []
        
        for method_name, results in [
            ('Stratified CV', stratified_results),
            ('Time Series CV', time_series_results),
            ('Stability Assessment', stability_results)
        ]:
            # Собираем метрики в зависимости от типа результатов
            if method_name == 'Stability Assessment':
                stats = results['stability_stats']
                for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
                    if f'{metric}_mean' in stats:
                        row = {
                            'Method': method_name,
                            'Metric': metric,
                            'Train Score': None,  # Для оценки стабильности нет отдельного Train Score
                            'Test Score': stats[f'{metric}_mean'],
                            'Std Dev': stats[f'{metric}_std'],
                            'CV (%)': stats[f'{metric}_cv'] * 100
                        }
                        comparison_data.append(row)
            else:
                for metric, values in results['metrics'].items():
                    row = {
                        'Method': method_name,
                        'Metric': metric,
                        'Train Score': values['train_mean'],
                        'Test Score': values['test_mean'],
                        'Std Dev': values['test_std'],
                        'CV (%)': (values['test_std'] / values['test_mean'] * 100 
                                  if values['test_mean'] != 0 else None)
                    }
                    comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Визуализация сравнения методов
        self._visualize_validation_comparison(comparison_df, model_name)
        
        # Сохранение результатов сравнения
        comparison_results = {
            'model': model_name,
            'stratified_cv': stratified_results,
            'time_series_cv': time_series_results,
            'stability_assessment': stability_results,
            'comparison_table': comparison_df.to_dict(orient='records')
        }
        
        # Сохранение в JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(self.results_dir, f'{model_name}_validation_comparison_{timestamp}.json')
        
        with open(results_path, 'w') as f:
            json.dump(comparison_results, f, indent=4)
        
        self.results[f'{model_name}_comparison'] = comparison_results
        
        return comparison_results
    
    def _visualize_validation_comparison(self, comparison_df, model_name):
        """
        Визуализирует сравнение методов валидации.
        
        Args:
            comparison_df (DataFrame): Данные для сравнения
            model_name (str): Имя модели
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        vis_dir = os.path.join(self.results_dir, 'plots')
        
        # Визуализация Test Score по методам и метрикам
        plt.figure(figsize=(12, 8))
        
        # Reshape data for grouped bar chart
        pivot_df = comparison_df.pivot_table(
            values='Test Score', 
            index='Metric', 
            columns='Method'
        )
        
        pivot_df.plot(kind='bar', ax=plt.gca())
        plt.title(f'{model_name} - Comparison of Validation Methods (Test Score)')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Method')
        plt.tight_layout()
        
        plt.savefig(os.path.join(vis_dir, f'{model_name}_validation_comparison_test_{timestamp}.png'))
        plt.close()
        
        # Визуализация Standard Deviation по методам и метрикам
        plt.figure(figsize=(12, 8))
        
        pivot_df = comparison_df.pivot_table(
            values='Std Dev', 
            index='Metric', 
            columns='Method'
        )
        
        pivot_df.plot(kind='bar', ax=plt.gca())
        plt.title(f'{model_name} - Comparison of Validation Methods (Standard Deviation)')
        plt.xlabel('Metric')
        plt.ylabel('Standard Deviation')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Method')
        plt.tight_layout()
        
        plt.savefig(os.path.join(vis_dir, f'{model_name}_validation_comparison_std_{timestamp}.png'))
        plt.close()
        
        # Визуализация Coefficient of Variation по методам и метрикам
        plt.figure(figsize=(12, 8))
        
        pivot_df = comparison_df.pivot_table(
            values='CV (%)', 
            index='Metric', 
            columns='Method'
        )
        
        pivot_df.plot(kind='bar', ax=plt.gca())
        plt.title(f'{model_name} - Comparison of Validation Methods (Coefficient of Variation)')
        plt.xlabel('Metric')
        plt.ylabel('CV (%)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Method')
        plt.tight_layout()
        
        plt.savefig(os.path.join(vis_dir, f'{model_name}_validation_comparison_cv_{timestamp}.png'))
        plt.close()

        
# Вспомогательная функция для клонирования модели
def clone_model(model):
    """
    Создает копию модели.
    
    Args:
        model: Исходная модель
        
    Returns:
        Model: Копия модели
    """
    from sklearn.base import clone
    try:
        return clone(model)
    except:
        # Для моделей из других библиотек может не работать clone
        # Пробуем сохранить и загрузить модель
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            joblib.dump(model, tmp_path)
            model_clone = joblib.load(tmp_path)
            os.unlink(tmp_path)
            return model_clone
        except:
            # Если и это не сработало, пробуем создать новый экземпляр с теми же параметрами
            if hasattr(model, '__class__') and hasattr(model, 'get_params'):
                return model.__class__(**model.get_params()) 