# Глава 2. Разработка моделей машинного обучения

## Выводы по Главе 2

Вторая глава настоящей работы была посвящена комплексному процессу разработки и оценки моделей машинного обучения, предназначенных для классификации клиентов компании Acoola по уровню их лояльности. Основной задачей являлось создание работающего прототипа системы, способной на основе имеющихся данных идентифицировать различные сегменты клиентов.

Работа в рамках данной главы включала несколько ключевых этапов:

1.  **Анализ и подготовка данных:** Были проанализированы исходные данные о транзакциях и взаимодействии клиентов с программой лояльности. Проведен обширный комплекс процедур по предобработке данных, включающий очистку от пропущенных значений и выбросов, трансформацию типов данных, кодирование категориальных признаков и нормализацию числовых переменных. Особое внимание было уделено конструированию признаков: выполнен RFM-анализ, разработаны расширенные признаки лояльности, проведена кластеризация клиентов и применен метод главных компонент (PCA) для обобщения информации. Кульминацией этого этапа стало формирование единого `enhanced_loyalty_score` и его последующая категоризация для получения целевой переменной `loyalty_target`.

2.  **Разработка моделей машинного обучения:** На подготовленном датасете был исследован ряд алгоритмов классификации, включая логистическую регрессию, случайный лес, метод опорных векторов и различные реализации градиентного бустинга (GradientBoostingClassifier, XGBoost, LightGBM). Был учтен дисбаланс классов в целевой переменной путем применения весовых коэффициентов при обучении.

3.  **Тестирование и оптимизация моделей:** Для наиболее перспективных моделей была проведена настройка гиперпараметров с использованием фреймворка Optuna и кросс-валидации, с целевой метрикой F1-macro. Проводился анализ важности признаков с помощью методов Permutation Importance, SHAP values и встроенных оценок моделей.

**Основные достигнутые результаты Главы 2:**

*   Разработан и реализован полный пайплайн обработки данных и моделирования, начиная от загрузки сырых данных и заканчивая сохранением обученной и оптимизированной модели.
*   В качестве наилучшей модели для классификации клиентов по уровню лояльности была выбрана **LightGBM**, которая после оптимизации гиперпараметров продемонстрировала на тестовой выборке метрику F1-macro, равную 1.0.
*   Были идентифицированы ключевые признаки, оказывающие наибольшее влияние на предсказание уровня лояльности. К ним относятся, в частности, `recency_ratio`, `monetary`, PCA-компоненты и `recency`.
*   Следует отметить, что достигнутые исключительно высокие метрики качества (F1-macro = 1.0) во многом обусловлены спецификой формирования двухкатегорийной целевой переменной на основе `enhanced_loyalty_score` и сильным дисбалансом классов. Это сделало задачу классификации относительно простой для современных ансамблевых методов. Данный аспект был принят во внимание, и было решено на текущем этапе зафиксировать эти результаты как отражение работы системы при текущей постановке задачи.

Результаты, полученные в данной главе, закладывают основу для практического применения разработанной модели. Созданный инструментарий позволяет не только классифицировать клиентов, но и получать представление о факторах, определяющих их лояльность, что является ценной информацией для формирования маркетинговых стратегий и персонализированных предложений. Подготовленная и сохраненная лучшая модель готова для интеграции в демонстрационное приложение, архитектура которого будет рассмотрена в следующей главе. 