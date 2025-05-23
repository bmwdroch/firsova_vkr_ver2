# Глава 2. Разработка моделей машинного обучения

## 2.1 Описание данных организации

Основу для исследования и разработки моделей машинного обучения составил набор данных, предоставленный компанией Acoola – одним из ведущих российских брендов детской одежды. Данные отражают покупательскую активность клиентов и их взаимодействие с программой лояльности компании.

**Основные характеристики набора данных:**

*   **Источник данных:** Внутренняя система учета компании Acoola.
*   **Наименование файла датасета:** `Concept202408.csv`.
*   **Объем данных:** Размер файла составляет приблизительно 113 МБ.
*   **Содержание данных:** Датасет включает агрегированную информацию о клиентах, их транзакциях, использовании бонусной программы и другие связанные метрики. Ключевыми атрибутами, использованными для анализа лояльности, являются:
    *   Данные о клиентах (идентификаторы, общая информация).
    *   Информация о покупках (даты, суммы, количество товаров).
    *   Данные по бонусной программе (начисление и списание бонусов, текущий баланс).
    *   Производные метрики, такие как частота покупок, средний чек, общая сумма покупок за период.
*   **Временной период:** Данные охватывают период активности клиентов за один месяц. *(Примечание: это предположение основано на упоминании в [TASK-002-5] об адаптации методов для данных за 1 месяц. Если период другой, это нужно скорректировать).*
*   **Конфиденциальность и подготовка данных:** Все персональные данные клиентов были предварительно анонимизированы или удалены из набора данных для обеспечения конфиденциальности и соответствия требованиям по защите персональных данных. Анализ проводился на обезличенных данных.

Данный набор данных был признан достаточным для проведения исследовательского анализа, выявления паттернов поведения клиентов и построения моделей классификации по уровню лояльности. Подробное описание полей датасета и их интерпретация были задокументированы в ходе предварительного этапа работы (см. `dataset_overview.md`, если применимо). 