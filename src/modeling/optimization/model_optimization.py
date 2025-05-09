import optuna
import sklearn.model_selection
import sklearn.metrics
from typing import Dict, Any, Callable, Tuple, Union
import pandas as pd

# Можно будет добавить сюда специфичные модели, если потребуется
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.ensemble import RandomForestClassifier

def optimize_hyperparameters(
    model_fn: Callable[..., Any], # Функция, создающая модель (например, RandomForestClassifier)
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_distributions: Callable[[optuna.trial.Trial], Dict[str, Any]],
    n_trials: int = 100,
    cv_folds: int = 5,
    scoring_metric: str = "f1_macro",
    random_state: int = 42
) -> Tuple[Dict[str, Any], float]:
    """
    Оптимизирует гиперпараметры для заданной модели с использованием Optuna.

    Args:
        model_fn: Функция, которая принимает гиперпараметры и возвращает экземпляр модели.
        X_train: Обучающие признаки.
        y_train: Целевая переменная для обучения.
        param_distributions: Функция, которая принимает optuna.trial.Trial и возвращает
                             словарь гиперпараметров для текущего испытания.
        n_trials: Количество испытаний Optuna.
        cv_folds: Количество фолдов для кросс-валидации.
        scoring_metric: Метрика для оптимизации (например, 'f1_macro', 'accuracy').
        random_state: Random state для воспроизводимости.

    Returns:
        Кортеж: (лучшие_гиперпараметры, лучший_результат_метрики)
    """

    def objective(trial: optuna.trial.Trial) -> float:
        # Генерация гиперпараметров для текущего испытания
        params = param_distributions(trial)
        
        # Создание модели с текущими гиперпараметрами
        model = model_fn(**params)

        # Кросс-валидация
        # Убедимся, что используется стратифицированная K-Fold для задач классификации
        cv = sklearn.model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Оценка модели
        # Optuna минимизирует, поэтому для метрик типа F1, accuracy нужно вернуть отрицательное значение
        # или использовать direction='maximize' в study.optimize
        scores = sklearn.model_selection.cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_metric)
        
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    print(f"Лучшие гиперпараметры: {best_params}")
    print(f"Лучшее значение метрики ({scoring_metric}): {best_value}")

    return best_params, best_value

# --- Функции для определения пространства поиска гиперпараметров --- #

def rf_param_distributions(trial: optuna.trial.Trial, use_class_weight: bool = True) -> Dict[str, Any]:
    """Определяет пространство поиска гиперпараметров для RandomForestClassifier."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 50, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        # "max_features": trial.suggest_float("max_features", 0.1, 1.0, log=True), # Можно добавить, если признаков много
    }
    if use_class_weight:
        params["class_weight"] = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
    return params

def xgb_param_distributions(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Определяет пространство поиска гиперпараметров для XGBClassifier."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 5, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True), # L2 regularization
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),   # L1 regularization
        # "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 100, log=True) # Для несбалансированных классов, если не используется другой метод
    }

def lgbm_param_distributions(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Определяет пространство поиска гиперпараметров для LGBMClassifier."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True), # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True), # L2 regularization
        # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        # "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]) # Для несбалансированных классов
    }

# Пример использования (потребуется раскомментировать и адаптировать, когда будет интеграция)
# if __name__ == '__main__':
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.datasets import make_classification

#     # Пример данных
#     X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
#     X_train_df = pd.DataFrame(X)
#     y_train_s = pd.Series(y)

#     # Определение функции для создания модели
#     def rf_model_fn(**params):
#         # Убедимся, что random_state передается, если он есть в params или используется глобальный
#         if 'random_state' not in params:
#             params['random_state'] = 42 
#         return RandomForestClassifier(**params)

#     # Определение пространства поиска гиперпараметров для RandomForest
#     def rf_param_distributions(trial: optuna.trial.Trial) -> Dict[str, Any]:
#         return {
#             "n_estimators": trial.suggest_int("n_estimators", 50, 300),
#             "max_depth": trial.suggest_int("max_depth", 3, 30),
#             "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
#             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
#             "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
#         }

#     print("Запуск оптимизации для RandomForestClassifier...")
#     best_rf_params, best_rf_score = optimize_hyperparameters(
#         model_fn=rf_model_fn,
#         X_train=X_train_df,
#         y_train=y_train_s,
#         param_distributions=rf_param_distributions,
#         n_trials=50, # Уменьшено для быстрого примера
#         cv_folds=3,
#         scoring_metric="f1_macro"
#     )
#     print(f"RandomForest - Лучшие параметры: {best_rf_params}, Лучший F1 macro: {best_rf_score}")

#     # Можно добавить аналогично для XGBoost, LightGBM
#     # ... 