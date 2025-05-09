import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json
from datetime import datetime
import itertools


class ModelEvaluator:
    """
    Класс для оценки и визуализации результатов моделей классификации.
    Включает функции для построения различных графиков, оценки качества
    классификации и генерации отчетов.
    """
    
    def __init__(self, models, results_dir='results'):
        """
        Инициализация оценщика моделей.
        
        Args:
            models (dict): Словарь с обученными моделями.
            results_dir (str): Директория для сохранения результатов.
        """
        self.models = models
        self.results_dir = results_dir
        self.evaluation_results = {}
        
        # Создаем директории, если они не существуют
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    def evaluate_models(self, X_test, y_test, class_names=None):
        """
        Оценивает все модели на тестовых данных.
        
        Args:
            X_test: Тестовые данные.
            y_test: Целевые значения.
            class_names (list): Имена классов.
            
        Returns:
            dict: Результаты оценки всех моделей.
        """
        for name, model in self.models.items():
            print(f"Оценка модели: {name}")
            y_pred = model.predict(X_test)
            
            try:
                y_proba = model.predict_proba(X_test)
            except:
                y_proba = None
            
            # Метрики качества
            accuracy = accuracy_score(y_test, y_pred)
            precision_macro = precision_score(y_test, y_pred, average='macro')
            recall_macro = recall_score(y_test, y_pred, average='macro')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # Полный отчет
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Матрица ошибок
            cm = confusion_matrix(y_test, y_pred)
            
            # Сохранение результатов
            self.evaluation_results[name] = {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'y_pred': y_pred.tolist(),
                'y_proba': y_proba.tolist() if y_proba is not None else None
            }
        
        return self.evaluation_results
    
    def get_best_model(self, metric='f1_macro'):
        """
        Возвращает лучшую модель по указанной метрике.
        
        Args:
            metric (str): Метрика для сравнения.
            
        Returns:
            tuple: (имя_лучшей_модели, значение_метрики)
        """
        best_score = -1
        best_model = None
        
        for name, results in self.evaluation_results.items():
            if metric in results and results[metric] > best_score:
                best_score = results[metric]
                best_model = name
        
        return best_model, best_score
    
    def plot_confusion_matrices(self, X_test, y_test, class_names=None, figsize=(15, 12)):
        """
        Строит матрицы ошибок для всех моделей.
        
        Args:
            X_test: Тестовые данные.
            y_test: Целевые значения.
            class_names (list): Имена классов.
            figsize (tuple): Размер фигуры.
            
        Returns:
            matplotlib.figure.Figure: Объект фигуры.
        """
        n_models = len(self.models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_models == 1:
            axes = np.array([axes])
        
        axes = axes.flatten()
        
        for i, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            if class_names is None:
                class_names = np.unique(y_test)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=axes[i])
            axes[i].set_title(f'Матрица ошибок: {name}')
            axes[i].set_xlabel('Предсказанные значения')
            axes[i].set_ylabel('Истинные значения')
        
        # Скрываем неиспользуемые субграфики
        for i in range(n_models, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Сохранение
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.results_dir, 'plots', f'confusion_matrices_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(self, X_test, y_test, figsize=(12, 8)):
        """
        Строит ROC-кривые для всех моделей.
        
        Args:
            X_test: Тестовые данные.
            y_test: Целевые значения.
            figsize (tuple): Размер фигуры.
            
        Returns:
            matplotlib.figure.Figure: Объект фигуры.
        """
        plt.figure(figsize=figsize)
        
        # Перекодирование меток для многоклассовой классификации
        classes = np.unique(y_test)
        n_classes = len(classes)
        
        if n_classes > 2:
            y_test_bin = label_binarize(y_test, classes=classes)
        else:
            y_test_bin = y_test
        
        for name, model in self.models.items():
            try:
                if n_classes > 2:
                    # Многоклассовая классификация
                    y_score = model.predict_proba(X_test)
                    
                    # Расчет ROC AUC для каждого класса
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    # Микро-усреднение
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    
                    plt.plot(fpr["micro"], tpr["micro"], label=f'{name} (AUC = {roc_auc["micro"]:.3f})')
                else:
                    # Бинарная классификация
                    y_score = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
            except (AttributeError, ValueError) as e:
                print(f"Не удалось построить ROC для модели {name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривые для моделей классификации')
        plt.legend(loc="lower right")
        
        # Сохранение
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.results_dir, 'plots', f'roc_curves_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curves(self, X_test, y_test, figsize=(12, 8)):
        """
        Строит кривые точности-полноты для всех моделей.
        
        Args:
            X_test: Тестовые данные.
            y_test: Целевые значения.
            figsize (tuple): Размер фигуры.
            
        Returns:
            matplotlib.figure.Figure: Объект фигуры.
        """
        plt.figure(figsize=figsize)
        
        # Перекодирование меток для многоклассовой классификации
        classes = np.unique(y_test)
        n_classes = len(classes)
        
        if n_classes > 2:
            y_test_bin = label_binarize(y_test, classes=classes)
        else:
            y_test_bin = y_test
        
        for name, model in self.models.items():
            try:
                if n_classes > 2:
                    # Многоклассовая классификация
                    y_score = model.predict_proba(X_test)
                    
                    # Расчет precision-recall для каждого класса
                    precision = dict()
                    recall = dict()
                    avg_precision = dict()
                    
                    for i in range(n_classes):
                        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
                        avg_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
                    
                    # Микро-усреднение
                    precision["micro"], recall["micro"], _ = precision_recall_curve(
                        y_test_bin.ravel(), y_score.ravel())
                    avg_precision["micro"] = average_precision_score(y_test_bin.ravel(), y_score.ravel())
                    
                    plt.plot(recall["micro"], precision["micro"], 
                             label=f'{name} (AP = {avg_precision["micro"]:.3f})')
                else:
                    # Бинарная классификация
                    y_score = model.predict_proba(X_test)[:, 1]
                    precision, recall, _ = precision_recall_curve(y_test, y_score)
                    avg_precision = average_precision_score(y_test, y_score)
                    plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})')
            except (AttributeError, ValueError) as e:
                print(f"Не удалось построить PR для модели {name}: {e}")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Полнота (Recall)')
        plt.ylabel('Точность (Precision)')
        plt.title('Кривые Precision-Recall для моделей классификации')
        plt.legend(loc="lower left")
        
        # Сохранение
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.results_dir, 'plots', f'precision_recall_curves_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_feature_importance(self, X, feature_names=None, top_n=20, figsize=(14, 10)):
        """
        Строит графики важности признаков для всех поддерживаемых моделей.
        
        Args:
            X: Данные.
            feature_names (list): Имена признаков.
            top_n (int): Количество наиболее важных признаков для отображения.
            figsize (tuple): Размер фигуры.
            
        Returns:
            matplotlib.figure.Figure: Объект фигуры.
        """
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Определение моделей, поддерживающих важность признаков
        models_with_importance = []
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                models_with_importance.append(name)
        
        n_models = len(models_with_importance)
        
        if n_models == 0:
            print("Ни одна модель не поддерживает оценку важности признаков")
            return None
        
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_models == 1:
            axes = np.array([axes])
        
        axes = axes.flatten()
        
        for i, name in enumerate(models_with_importance):
            model = self.models[name]
            
            # Получение важности признаков
            importance = None
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            
            # Создание DataFrame и сортировка
            features_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            features_df = features_df.sort_values('importance', ascending=False).head(top_n)
            
            # Визуализация
            sns.barplot(x='importance', y='feature', data=features_df, ax=axes[i])
            axes[i].set_title(f'Важность признаков: {name}')
            axes[i].set_xlabel('Важность')
            axes[i].set_ylabel('Признак')
        
        # Скрываем неиспользуемые субграфики
        for i in range(n_models, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Сохранение
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.results_dir, 'plots', f'feature_importance_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(self, metrics=None, figsize=(14, 8)):
        """
        Строит сравнительную диаграмму метрик для всех моделей.
        
        Args:
            metrics (list): Список метрик для сравнения.
            figsize (tuple): Размер фигуры.
            
        Returns:
            matplotlib.figure.Figure: Объект фигуры.
        """
        if metrics is None:
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Создание DataFrame для сравнения
        comparison_data = []
        for name, results in self.evaluation_results.items():
            row = {'model': name}
            for metric in metrics:
                if metric in results:
                    row[metric] = results[metric]
            comparison_data.append(row)
        
        if not comparison_data:
            print("Нет данных для сравнения")
            return None
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Визуализация
        fig, ax = plt.subplots(figsize=figsize)
        comparison_df.set_index('model').plot(kind='bar', ax=ax)
        plt.title('Сравнение метрик качества моделей')
        plt.ylabel('Значение метрики')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Метрика')
        
        # Сохранение
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.results_dir, 'plots', f'metrics_comparison_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_class_distribution(self, y_true, y_pred_dict=None, figsize=(14, 8)):
        """
        Строит распределение классов для истинных и предсказанных значений.
        
        Args:
            y_true: Истинные значения.
            y_pred_dict (dict): Словарь с предсказаниями разных моделей.
            figsize (tuple): Размер фигуры.
            
        Returns:
            matplotlib.figure.Figure: Объект фигуры.
        """
        if y_pred_dict is None:
            y_pred_dict = {}
            for name, model in self.models.items():
                y_pred_dict[name] = model.predict(X_test)
        
        # Подсчет частот классов
        true_counts = pd.Series(y_true).value_counts().sort_index()
        classes = true_counts.index.tolist()
        
        # Создание DataFrame для сравнения
        counts_df = pd.DataFrame({'Истинные': true_counts})
        for name, y_pred in y_pred_dict.items():
            pred_counts = pd.Series(y_pred).value_counts().sort_index()
            counts_df[name] = pred_counts
        
        # Визуализация
        fig, ax = plt.subplots(figsize=figsize)
        counts_df.plot(kind='bar', ax=ax)
        plt.title('Распределение классов: истинные vs предсказанные')
        plt.ylabel('Количество объектов')
        plt.xlabel('Класс')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Сохранение
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.results_dir, 'plots', f'class_distribution_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_evaluation_report(self, filename=None):
        """
        Генерирует отчет по оценке моделей.
        
        Args:
            filename (str): Имя файла для сохранения отчета.
            
        Returns:
            str: Текст отчета.
        """
        if not self.evaluation_results:
            return "Нет данных для формирования отчета"
        
        report = []
        report.append("# Отчет по оценке моделей классификации")
        report.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Сравнение метрик качества")
        
        # Создание таблицы метрик
        metrics_table = pd.DataFrame([
            {
                'Модель': name, 
                'Accuracy': results.get('accuracy', '-'),
                'Precision (macro)': results.get('precision_macro', '-'),
                'Recall (macro)': results.get('recall_macro', '-'),
                'F1 (macro)': results.get('f1_macro', '-')
            }
            for name, results in self.evaluation_results.items()
        ])
        
        report.append(metrics_table.to_markdown(index=False))
        
        # Лучшая модель
        best_model, best_score = self.get_best_model(metric='f1_macro')
        report.append(f"\n## Лучшая модель по F1-мере")
        report.append(f"**{best_model}** с F1-мерой = {best_score:.4f}")
        
        # Детали по каждой модели
        report.append("\n## Детализация по моделям")
        
        for name, results in self.evaluation_results.items():
            report.append(f"\n### {name}")
            if 'classification_report' in results:
                # Преобразование classification_report в таблицу
                class_report = results['classification_report']
                report_df = pd.DataFrame({
                    'Класс': list(class_report.keys())[:-3],  # Исключаем accuracy, macro avg, weighted avg
                    'Precision': [class_report[cls]['precision'] for cls in list(class_report.keys())[:-3]],
                    'Recall': [class_report[cls]['recall'] for cls in list(class_report.keys())[:-3]],
                    'F1-Score': [class_report[cls]['f1-score'] for cls in list(class_report.keys())[:-3]],
                    'Support': [class_report[cls]['support'] for cls in list(class_report.keys())[:-3]]
                })
                report.append(report_df.to_markdown(index=False))
            
            report.append("\n#### Средние метрики")
            for avg in ['macro avg', 'weighted avg']:
                if avg in results.get('classification_report', {}):
                    avg_metrics = results['classification_report'][avg]
                    report.append(f"- **{avg}**: "
                                 f"Precision = {avg_metrics['precision']:.4f}, "
                                 f"Recall = {avg_metrics['recall']:.4f}, "
                                 f"F1-score = {avg_metrics['f1-score']:.4f}")
        
        # Сохранение отчета
        report_text = "\n".join(report)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.results_dir, f'evaluation_report_{timestamp}.md')
        else:
            filename = os.path.join(self.results_dir, filename)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def save_evaluation_results(self, filename=None):
        """
        Сохраняет результаты оценки в файл JSON.
        
        Args:
            filename (str): Имя файла.
            
        Returns:
            str: Путь к сохраненному файлу.
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"evaluation_results_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Преобразование результатов в сериализуемый формат
        serializable_results = {}
        for model_name, model_results in self.evaluation_results.items():
            serializable_model = {}
            for key, value in model_results.items():
                if isinstance(value, (dict, list, int, float, str, bool)) or value is None:
                    serializable_model[key] = value
                elif isinstance(value, np.ndarray):
                    serializable_model[key] = value.tolist()
                else:
                    serializable_model[key] = str(value)
            serializable_results[model_name] = serializable_model
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        return filepath 