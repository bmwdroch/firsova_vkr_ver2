import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import json

class ModelTrainer:
    """
    Класс для обучения и оценки моделей классификации клиентов по уровню лояльности.
    Поддерживает различные типы моделей, оценку на тестовой выборке, кросс-валидацию,
    подбор гиперпараметров и создание ансамблевых моделей.
    """
    
    def __init__(self, models_dir='models', results_dir='results'):
        """
        Инициализация тренера моделей.
        
        Args:
            models_dir (str): Директория для сохранения моделей.
            results_dir (str): Директория для сохранения результатов оценки.
        """
        self.models = {}
        self.results = {}
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Создаем директории, если они не существуют
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
    def add_base_models(self, class_weights=None, model_custom_params: dict = None):
        """
        Добавляет базовые модели классификации.
        
        Args:
            class_weights (dict, optional): Веса классов для учета несбалансированности.
            model_custom_params (dict, optional): Словарь с пользовательскими гиперпараметрами для моделей.
                                                 Ключ - имя модели, значение - словарь гиперпараметров.
        """
        if model_custom_params is None:
            model_custom_params = {}

        # Logistic Regression
        lr_params = {
            'max_iter': 1000, 
            'class_weight': class_weights, 
            'random_state': 42,
            **model_custom_params.get('logistic_regression', {})
        }
        self.models['logistic_regression'] = LogisticRegression(**lr_params)
        
        # Random Forest
        rf_params = {
            'n_estimators': 100,
            'class_weight': class_weights,
            'random_state': 42,
            **model_custom_params.get('random_forest', {})
        }
        self.models['random_forest'] = RandomForestClassifier(**rf_params)
        
        # Gradient Boosting
        gb_params = {
            'n_estimators': 100,
            'random_state': 42,
            **model_custom_params.get('gradient_boosting', {})
        }
        # GradientBoostingClassifier не принимает class_weight напрямую в конструкторе,
        # но его можно учесть через sample_weight при обучении, если потребуется
        self.models['gradient_boosting'] = GradientBoostingClassifier(**gb_params)
        
        # XGBoost
        xgb_params = {
            'n_estimators': 100,
            'use_label_encoder': False, # deprecated, use enable_categorical=True for newer versions
            'eval_metric': 'mlogloss',
            'random_state': 42,
            **model_custom_params.get('xgboost', {})
        }
        # XGBoost может использовать scale_pos_weight для балансировки или sample_weight
        self.models['xgboost'] = XGBClassifier(**xgb_params)
        
        # LightGBM
        lgbm_params = {
            'n_estimators': 100,
            'class_weight': class_weights, # Поддерживается
            'min_child_samples': 20, # Пример параметра по умолчанию
            'min_split_gain': 0.1,   # Пример параметра по умолчанию
            'random_state': 42,
            'verbosity': -1,
            **model_custom_params.get('lightgbm', {})
        }
        self.models['lightgbm'] = LGBMClassifier(**lgbm_params)
        
        # SVM
        svm_params = {
            'probability': True,
            'class_weight': class_weights,
            'random_state': 42,
            **model_custom_params.get('svm', {})
        }
        self.models['svm'] = SVC(**svm_params)
    
    def add_custom_model(self, name, model_class, hyperparams: dict = None):
        """
        Добавляет пользовательскую модель с возможностью указания гиперпараметров.
        
        Args:
            name (str): Имя модели.
            model_class: Класс модели (например, RandomForestClassifier).
            hyperparams (dict, optional): Словарь гиперпараметров для модели.
        """
        if hyperparams is None:
            hyperparams = {}
        # Убедимся, что random_state передается, если он применим и не задан
        # Это общая логика, специфичные модели могут требовать другой обработки
        if 'random_state' not in hyperparams and hasattr(model_class(), 'random_state'):
            hyperparams['random_state'] = 42
            
        self.models[name] = model_class(**hyperparams)
    
    def create_voting_ensemble(self, estimators=None, voting='soft', hyperparams: dict = None):
        """
        Создает ансамблевую модель голосования.
        
        Args:
            estimators (list, optional): Список кортежей (имя, модель).
                Если None, используются все базовые модели.
            voting (str): Тип голосования ('hard' или 'soft').
            hyperparams (dict, optional): Словарь гиперпараметров для ансамблевой модели.
        """
        if estimators is None:
            estimators = [(name, model) for name, model in self.models.items()]
        
        if hyperparams is None:
            hyperparams = {}
        
        self.models['voting_ensemble'] = VotingClassifier(
            estimators=estimators,
            voting=voting,
            **hyperparams
        )
    
    def create_stacking_ensemble(self, estimators=None, final_estimator=None, hyperparams: dict = None):
        """
        Создает ансамблевую модель стекинга.
        
        Args:
            estimators (list, optional): Список кортежей (имя, модель).
                Если None, используются все базовые модели.
            final_estimator: Модель второго уровня.
                Если None, используется логистическая регрессия.
            hyperparams (dict, optional): Словарь гиперпараметров для ансамблевой модели.
        """
        if estimators is None:
            estimators = [(name, model) for name, model in self.models.items()]
        
        if final_estimator is None:
            final_estimator = LogisticRegression(max_iter=1000)
        
        if hyperparams is None:
            hyperparams = {}
        
        self.models['stacking_ensemble'] = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            **hyperparams
        )
    
    def train_model(self, name, X_train, y_train, sample_weight=None):
        """
        Обучает модель.
        
        Args:
            name (str): Имя модели.
            X_train: Обучающие данные.
            y_train: Целевые значения.
            sample_weight (array-like, optional): Веса для образцов.
                                                 Используется для моделей, которые это поддерживают,
                                                 например, если class_weights не применяется напрямую.
            
        Returns:
            Обученная модель.
        """
        start_time = time.time()
        model = self.models[name]

        fit_params = {}
        if sample_weight is not None:
            # Некоторые модели принимают sample_weight в fit
            # Проверяем, есть ли такой параметр у метода fit
            import inspect
            sig = inspect.signature(model.fit)
            if 'sample_weight' in sig.parameters:
                fit_params['sample_weight'] = sample_weight
            else:
                print(f"Предупреждение: Модель {name} не поддерживает sample_weight в fit.")

        if not fit_params: # Если sample_weight не был добавлен
             model.fit(X_train, y_train)
        else:
             model.fit(X_train, y_train, **fit_params)

        training_time = time.time() - start_time
        
        self.results.setdefault(name, {})['training_time'] = training_time
        
        return model
    
    def train_all_models(self, X_train, y_train, class_weights_dict=None):
        """
        Обучает все модели.
        
        Args:
            X_train: Обучающие данные.
            y_train: Целевые значения.
            class_weights_dict (dict, optional): Словарь весов классов для sample_weight. 
                                                Ключ - класс, значение - вес.
        """
        sample_weights_train = None
        if class_weights_dict is not None and isinstance(y_train, (pd.Series, np.ndarray)):
            # Преобразуем y_train в массив numpy, если это pd.Series, для map
            y_train_array = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
            # Убедимся, что y_train_array - это 1D массив целых чисел (классов)
            if y_train_array.ndim == 1 and np.issubdtype(y_train_array.dtype, np.integer):
                sample_weights_train = np.array([class_weights_dict.get(cls, 1.0) for cls in y_train_array])
            else:
                print("Предупреждение: y_train должен быть 1D массивом целых чисел для использования class_weights_dict с sample_weight.")

        for name in self.models:
            print(f"Обучение модели: {name}")
            self.train_model(name, X_train, y_train, sample_weight=sample_weights_train)
    
    def evaluate_model(self, name, X_test, y_test):
        """
        Оценивает модель на тестовых данных.
        
        Args:
            name (str): Имя модели.
            X_test: Тестовые данные.
            y_test: Целевые значения.
            
        Returns:
            dict: Результаты оценки.
        """
        model = self.models[name]
        
        # Предсказание классов и вероятностей
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        try:
            y_proba = model.predict_proba(X_test)
        except AttributeError:
            y_proba = None
        
        # Расчет метрик
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Сохранение результатов
        results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'prediction_time': prediction_time,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.results.setdefault(name, {}).update(results)
        
        return results
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Оценивает все модели на тестовых данных.
        
        Args:
            X_test: Тестовые данные.
            y_test: Целевые значения.
            
        Returns:
            dict: Результаты оценки всех моделей.
        """
        all_results = {}
        
        for name in self.models:
            print(f"Оценка модели: {name}")
            results = self.evaluate_model(name, X_test, y_test)
            all_results[name] = results
        
        return all_results
    
    def compare_models(self, metric='f1_macro'):
        """
        Сравнивает модели по заданной метрике.
        
        Args:
            metric (str): Метрика для сравнения.
            
        Returns:
            pandas.DataFrame: Сравнительная таблица моделей.
        """
        comparison = []
        
        for name, results in self.results.items():
            if metric in results:
                row = {
                    'model': name,
                    metric: results[metric],
                    'accuracy': results.get('accuracy', float('nan')),
                    'training_time': results.get('training_time', float('nan')),
                    'prediction_time': results.get('prediction_time', float('nan'))
                }
                comparison.append(row)
        
        df = pd.DataFrame(comparison)
        df = df.sort_values(by=metric, ascending=False)
        
        return df
    
    def plot_confusion_matrix(self, name, X_test, y_test, cmap='Blues'):
        """
        Строит матрицу ошибок для модели.
        
        Args:
            name (str): Имя модели.
            X_test: Тестовые данные.
            y_test: Целевые значения.
            cmap (str): Цветовая схема.
            
        Returns:
            matplotlib.figure.Figure: Объект фигуры.
        """
        model = self.models[name]
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        classes = np.unique(y_test)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Истинные значения')
        plt.title(f'Матрица ошибок для модели {name}')
        
        return plt.gcf()
    
    def plot_roc_curves(self, X_test, y_test):
        """
        Строит ROC-кривые для всех моделей.
        
        Args:
            X_test: Тестовые данные.
            y_test: Целевые значения.
            
        Returns:
            matplotlib.figure.Figure: Объект фигуры.
        """
        plt.figure(figsize=(12, 10))
        
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
        
        return plt.gcf()
    
    def plot_precision_recall_curves(self, X_test, y_test):
        """
        Строит кривые точности-полноты для всех моделей.
        
        Args:
            X_test: Тестовые данные.
            y_test: Целевые значения.
            
        Returns:
            matplotlib.figure.Figure: Объект фигуры.
        """
        plt.figure(figsize=(12, 10))
        
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
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Кривые Precision-Recall для моделей классификации')
        plt.legend(loc="lower left")
        
        return plt.gcf()
    
    def tune_hyperparameters(self, name, X_train, y_train, param_grid, cv=5, method='grid', n_iter=10, scoring='f1_macro'):
        """
        Настройка гиперпараметров модели.
        
        Args:
            name (str): Имя модели.
            X_train: Обучающие данные.
            y_train: Целевые значения.
            param_grid (dict): Сетка параметров.
            cv (int): Количество фолдов в кросс-валидации.
            method (str): Метод поиска ('grid' или 'random').
            n_iter (int): Количество итераций для random search.
            scoring (str): Метрика для оптимизации.
            
        Returns:
            Модель с оптимальными гиперпараметрами.
        """
        model = self.models[name]
        
        if method == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
            )
        else:  # random
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, verbose=1, random_state=42
            )
        
        start_time = time.time()
        search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        # Сохранение результатов
        tuning_results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'tuning_time': tuning_time
        }
        self.results.setdefault(name, {})['tuning'] = tuning_results
        
        # Обновление модели
        self.models[name] = search.best_estimator_
        
        return search.best_estimator_
    
    def cross_validate_model(self, name, X, y, cv=5, scoring=None):
        """
        Проводит кросс-валидацию модели.
        
        Args:
            name (str): Имя модели.
            X: Данные.
            y: Целевые значения.
            cv (int): Количество фолдов.
            scoring (str, list): Метрики для оценки.
            
        Returns:
            dict: Результаты кросс-валидации.
        """
        if scoring is None:
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        model = self.models[name]
        
        start_time = time.time()
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1
        )
        cv_time = time.time() - start_time
        
        # Обработка результатов
        cv_metrics = {}
        for metric in scoring:
            train_scores = cv_results[f'train_{metric}']
            test_scores = cv_results[f'test_{metric}']
            cv_metrics[f'{metric}_train_mean'] = train_scores.mean()
            cv_metrics[f'{metric}_train_std'] = train_scores.std()
            cv_metrics[f'{metric}_test_mean'] = test_scores.mean()
            cv_metrics[f'{metric}_test_std'] = test_scores.std()
        
        cv_metrics['cv_time'] = cv_time
        
        # Сохранение результатов
        self.results.setdefault(name, {})['cross_validation'] = cv_metrics
        
        return cv_metrics
    
    def plot_feature_importance(self, name, X, feature_names=None):
        """
        Визуализирует важность признаков для модели.
        
        Args:
            name (str): Имя модели.
            X: Данные.
            feature_names (list, optional): Имена признаков.
            
        Returns:
            matplotlib.figure.Figure: Объект фигуры.
        """
        model = self.models[name]
        
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Получение важности признаков (если доступно)
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            print(f"Модель {name} не поддерживает оценку важности признаков")
            return None
        
        # Создание DataFrame и сортировка
        features_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        features_df = features_df.sort_values('importance', ascending=False)
        
        # Визуализация
        plt.figure(figsize=(12, max(6, len(features_df) * 0.3)))
        sns.barplot(x='importance', y='feature', data=features_df)
        plt.title(f'Важность признаков для модели {name}')
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_model(self, name, filename=None):
        """
        Сохраняет модель в файл.
        
        Args:
            name (str): Имя модели.
            filename (str, optional): Имя файла.
                Если None, генерируется автоматически.
            
        Returns:
            str: Путь к сохраненному файлу.
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{name}_{timestamp}.joblib"
        
        filepath = os.path.join(self.models_dir, filename)
        
        joblib.dump(self.models[name], filepath)
        
        return filepath
    
    def save_all_models(self):
        """
        Сохраняет все модели в файлы.
        
        Returns:
            dict: Словарь с путями к сохраненным файлам.
        """
        paths = {}
        
        for name in self.models:
            paths[name] = self.save_model(name)
        
        return paths
    
    def save_results(self, filename=None):
        """
        Сохраняет результаты оценки моделей в файл JSON.
        
        Args:
            filename (str, optional): Имя файла.
                Если None, генерируется автоматически.
            
        Returns:
            str: Путь к сохраненному файлу.
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"model_results_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Преобразование результатов в сериализуемый формат
        serializable_results = {}
        for model_name, model_results in self.results.items():
            serializable_model = {}
            for key, value in model_results.items():
                if isinstance(value, dict):
                    serializable_model[key] = value
                elif isinstance(value, (int, float, str, bool, list, tuple)) or value is None:
                    serializable_model[key] = value
                else:
                    serializable_model[key] = str(value)
            serializable_results[model_name] = serializable_model
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        return filepath
    
    def load_model(self, filepath):
        """
        Загружает модель из файла.
        
        Args:
            filepath (str): Путь к файлу.
            
        Returns:
            Model: Загруженная модель.
        """
        model = joblib.load(filepath)
        
        # Определение имени модели из файла
        basename = os.path.basename(filepath)
        name = basename.split('_')[0]
        
        self.models[name] = model
        
        return model


def get_hyperparameter_grids():
    """
    Возвращает сетки гиперпараметров для различных моделей.
    
    Returns:
        dict: Словарь с сетками гиперпараметров.
    """
    param_grids = {
        'logistic_regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'class_weight': [None, 'balanced']
        },
        'random_forest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.7, 0.8, 0.9, 1.0]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        },
        'lightgbm': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, -1],
            'num_leaves': [31, 63, 127],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly'],
            'class_weight': [None, 'balanced']
        }
    }
    
    return param_grids 