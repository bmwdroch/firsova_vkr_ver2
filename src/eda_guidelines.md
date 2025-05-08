# Рекомендации по исследовательскому анализу данных (EDA)

## Введение

Данный документ содержит рекомендации по проведению исследовательского анализа данных для проекта классификации клиентов компании Acoola по уровню лояльности. Документ включает пошаговый план анализа, перечень ключевых исследуемых переменных и рекомендуемые методы визуализации и статистического анализа.

## Исходные данные

Датасет `Concept202408.csv` содержит информацию о клиентах компании Acoola, их покупках, использовании бонусной программы и других метриках. Размер файла - около 113 МБ. Датасет включает следующие основные категории данных:

### Информация о клиенте
- **Клиент** – ID клиента
- **Пол** – пол клиента
- **Точка продаж где оформлена карта** – место оформления бонусной карты
- **Дата оформления карты** – дата регистрации в программе лояльности
- **Дата первого чека** – дата первой покупки
- **Дата последнего чека** – дата последней покупки

### Информация о покупках
- **Дата покупки** - дата совершения покупки
- **Точка продаж** – место совершения покупки
- **Название товара** – наименование купленного товара
- **Cумма покупки** – сумма покупки в руб.
- **Количество** – количество наименований товара в одной покупке

### Скидки и бонусы
- **Скидка внешняя** – скидка на покупку товаров, в руб.
- **Начислено бонусов** – количество начисленных бонусов, в руб.
- **Списано бонусов** – количество списанных бонусов, в руб.
- **Баланс накопленный** – количество накопленных бонусов на дату покупки, в руб.
- **Баланс подарочный** – количество бонусов, начисленных по акциям, в руб.

### Аналитические показатели
- **Покупок, в днях** - количество дней с покупками
- **Частота, раз/мес** – частота покупок в месяц
- **Средняя сумма покупок** - средний чек клиента, в руб.

## Пошаговый план EDA

### 1. Загрузка и первичный анализ структуры данных

1.1. Загрузка данных и проверка размерности датасета:
```python
import pandas as pd
df = pd.read_csv('../dataset/Concept202408.csv')
print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
```

1.2. Анализ типов данных и наличия пропущенных значений:
```python
# Проверка типов данных
df.info()

# Анализ пропущенных значений
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Кол-во пропущенных': missing_values,
    'Процент пропущенных': missing_percent
})
missing_df = missing_df[missing_df['Кол-во пропущенных'] > 0].sort_values('Процент пропущенных', ascending=False)
print(missing_df)
```

1.3. Преобразование дат и проверка диапазона данных:
```python
# Преобразование дат
date_columns = ['Дата покупки', 'Дата оформления карты', 'Дата первого чека', 'Дата последнего чека']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

# Проверка диапазона дат
if 'Дата покупки' in df.columns:
    print(f"Период данных: с {df['Дата покупки'].min()} по {df['Дата покупки'].max()}")
    print(f"Продолжительность: {(df['Дата покупки'].max() - df['Дата покупки'].min()).days} дней")
```

### 2. Исследование распределений основных переменных

2.1. Базовая статистика числовых переменных:
```python
df.describe().T
```

2.2. Анализ распределения категориальных переменных:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Пол клиентов
if 'Пол' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Пол', data=df)
    plt.title('Распределение клиентов по полу')
    plt.ylabel('Количество клиентов')
    plt.show()

# Точки продаж
if 'Точка продаж' in df.columns:
    top_locations = df['Точка продаж'].value_counts().head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_locations.values, y=top_locations.index)
    plt.title('Топ-10 точек продаж')
    plt.xlabel('Количество покупок')
    plt.tight_layout()
    plt.show()
```

2.3. Анализ распределения числовых переменных:
```python
# Выбор ключевых числовых переменных
numeric_vars = ['Cумма покупки', 'Начислено бонусов', 'Списано бонусов', 
                'Средняя сумма покупок', 'Частота, раз/мес']

# Анализ распределений
for var in numeric_vars:
    if var in df.columns:
        plt.figure(figsize=(14, 6))
        
        # Гистограмма
        plt.subplot(1, 2, 1)
        sns.histplot(df[var].dropna(), kde=True)
        plt.title(f'Распределение: {var}')
        
        # Boxplot для выявления выбросов
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[var].dropna())
        plt.title(f'Диаграмма размаха: {var}')
        
        plt.tight_layout()
        plt.show()
```

### 3. Агрегация данных на уровне клиента

3.1. Группировка данных по клиентам:
```python
# Группировка данных по клиентам
if 'Клиент' in df.columns:
    customer_df = df.groupby('Клиент').agg({
        'Cумма покупки': ['count', 'sum', 'mean'],
        'Дата покупки': ['min', 'max'],
        'Начислено бонусов': 'sum',
        'Списано бонусов': 'sum'
    })
    
    # Переименование колонок
    customer_df.columns = ['_'.join(col).strip() for col in customer_df.columns.values]
    customer_df.reset_index(inplace=True)
    
    # Расчет recency (для последнего дня в датасете)
    if 'Дата покупки_max' in customer_df.columns:
        last_date = df['Дата покупки'].max()
        customer_df['recency'] = (last_date - customer_df['Дата покупки_max']).dt.days
    
    # Переименование колонок для удобства
    customer_df = customer_df.rename(columns={
        'Cумма покупки_count': 'frequency',
        'Cумма покупки_sum': 'monetary',
        'Cумма покупки_mean': 'avg_purchase',
        'Начислено бонусов_sum': 'total_bonus_earned',
        'Списано бонусов_sum': 'total_bonus_used'
    })
    
    # Базовая статистика по клиентам
    print(customer_df.describe())
```

### 4. Анализ RFM-метрик

4.1. Расчет и анализ RFM-метрик:
```python
# Если у нас есть агрегированные данные по клиентам
if 'customer_df' in locals():
    # Создаем квантили для R, F, M
    quantiles = customer_df[['recency', 'frequency', 'monetary']].quantile([0.25, 0.5, 0.75]).to_dict()
    
    # Функции для присвоения RFM-оценок
    def r_score(x):
        if x <= quantiles['recency'][0.25]:
            return 4
        elif x <= quantiles['recency'][0.5]:
            return 3
        elif x <= quantiles['recency'][0.75]:
            return 2
        else:
            return 1
    
    def fm_score(x, metric):
        if x <= quantiles[metric][0.25]:
            return 1
        elif x <= quantiles[metric][0.5]:
            return 2
        elif x <= quantiles[metric][0.75]:
            return 3
        else:
            return 4
    
    # Присвоение RFM-оценок
    customer_df['R'] = customer_df['recency'].apply(r_score)
    customer_df['F'] = customer_df['frequency'].apply(lambda x: fm_score(x, 'frequency'))
    customer_df['M'] = customer_df['monetary'].apply(lambda x: fm_score(x, 'monetary'))
    
    # Создание RFM-группы и общего балла
    customer_df['RFM_Group'] = customer_df['R'].astype(str) + customer_df['F'].astype(str) + customer_df['M'].astype(str)
    customer_df['RFM_Score'] = customer_df['R'] + customer_df['F'] + customer_df['M']
    
    # Определение сегментов лояльности
    def loyalty_segment(score):
        if score >= 10:
            return 'Высоколояльные'
        elif score >= 8:
            return 'Умеренно лояльные'
        elif score >= 6:
            return 'Низколояльные'
        elif score >= 4:
            return 'Потенциально лояльные'
        else:
            return 'Группа оттока'
    
    customer_df['Loyalty_Segment'] = customer_df['RFM_Score'].apply(loyalty_segment)
    
    # Визуализация сегментов
    plt.figure(figsize=(12, 6))
    segment_order = ['Высоколояльные', 'Умеренно лояльные', 'Низколояльные', 'Потенциально лояльные', 'Группа оттока']
    sns.countplot(x='Loyalty_Segment', data=customer_df, order=segment_order)
    plt.title('Распределение клиентов по сегментам лояльности')
    plt.xlabel('Сегмент лояльности')
    plt.ylabel('Количество клиентов')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Статистика по сегментам
    segment_stats = customer_df.groupby('Loyalty_Segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'Клиент': 'count'
    }).rename(columns={'Клиент': 'Count'}).reset_index()
    
    print("Статистика по сегментам лояльности:")
    print(segment_stats)
```

### 5. Исследование взаимосвязей между переменными

5.1. Корреляционный анализ:
```python
# Корреляция между числовыми переменными
if 'customer_df' in locals():
    corr_cols = ['recency', 'frequency', 'monetary', 'avg_purchase', 
                 'total_bonus_earned', 'total_bonus_used']
    corr_cols = [col for col in corr_cols if col in customer_df.columns]
    
    correlation = customer_df[corr_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Корреляция между показателями лояльности')
    plt.tight_layout()
    plt.show()
```

5.2. Анализ взаимосвязи между метриками:
```python
# Взаимосвязь между частотой и суммой покупок
plt.figure(figsize=(10, 6))
sns.scatterplot(x='frequency', y='monetary', data=customer_df)
plt.title('Взаимосвязь между частотой и суммой покупок')
plt.xlabel('Количество покупок')
plt.ylabel('Общая сумма покупок, руб.')
plt.show()

# Взаимосвязь между суммой покупок и начисленными бонусами
if 'total_bonus_earned' in customer_df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='monetary', y='total_bonus_earned', data=customer_df)
    plt.title('Взаимосвязь между суммой покупок и начисленными бонусами')
    plt.xlabel('Общая сумма покупок, руб.')
    plt.ylabel('Начислено бонусов, руб.')
    plt.show()
```

### 6. Временной анализ

6.1. Анализ покупок по времени:
```python
# Анализ покупок по дням недели
if 'Дата покупки' in df.columns:
    df['day_of_week'] = df['Дата покупки'].dt.day_name()
    
    plt.figure(figsize=(12, 6))
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.countplot(x='day_of_week', data=df, order=order)
    plt.title('Распределение покупок по дням недели')
    plt.xlabel('День недели')
    plt.ylabel('Количество покупок')
    plt.tight_layout()
    plt.show()

# Анализ покупок по дням месяца
if 'Дата покупки' in df.columns:
    df['day_of_month'] = df['Дата покупки'].dt.day
    
    plt.figure(figsize=(14, 6))
    day_counts = df.groupby('day_of_month').size()
    sns.barplot(x=day_counts.index, y=day_counts.values)
    plt.title('Распределение покупок по дням месяца')
    plt.xlabel('День месяца')
    plt.ylabel('Количество покупок')
    plt.tight_layout()
    plt.show()
```

### 7. Анализ товарных категорий

7.1. Анализ популярности товаров:
```python
# Топ-10 самых популярных товаров
if 'Название товара' in df.columns:
    top_products = df['Название товара'].value_counts().head(10)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_products.values, y=top_products.index)
    plt.title('Топ-10 самых популярных товаров')
    plt.xlabel('Количество покупок')
    plt.ylabel('Название товара')
    plt.tight_layout()
    plt.show()

# Средний чек по товарным категориям
if 'Название товара' in df.columns and 'Cумма покупки' in df.columns:
    product_avg = df.groupby('Название товара')['Cумма покупки'].mean().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=product_avg.values, y=product_avg.index)
    plt.title('Топ-10 товаров по среднему чеку')
    plt.xlabel('Средний чек, руб.')
    plt.ylabel('Название товара')
    plt.tight_layout()
    plt.show()
```

## Рекомендации по интерпретации результатов

При анализе результатов EDA рекомендуется обратить внимание на следующие аспекты:

1. **Распределение клиентов по сегментам лояльности**:
   - Какая доля клиентов относится к высоколояльным и умеренно лояльным сегментам?
   - Какие характеристики имеют клиенты из каждого сегмента?

2. **Ключевые факторы лояльности**:
   - Какие переменные наиболее сильно коррелируют с сегментом лояльности?
   - Какие закономерности наблюдаются в покупательском поведении лояльных клиентов?

3. **Временные паттерны**:
   - Есть ли сезонность или цикличность в покупках клиентов?
   - Как меняется активность клиентов разных сегментов лояльности в течение месяца?

4. **Товарные предпочтения**:
   - Есть ли различия в товарных предпочтениях между сегментами лояльности?
   - Какие товары приобретают высоколояльные клиенты?

5. **Использование бонусной программы**:
   - Как соотносится накопление и использование бонусов с уровнем лояльности?
   - Есть ли связь между активностью использования бонусов и частотой покупок?

## Заключение

Исследовательский анализ данных является ключевым этапом для понимания структуры и особенностей данных перед разработкой модели классификации клиентов по уровню лояльности. Результаты EDA позволят выявить наиболее значимые факторы лояльности и определить направления для дальнейшей работы над моделью.

При проведении анализа рекомендуется использовать Python-скрипты для исследования данных и визуализации результатов. Все промежуточные выводы и наблюдения следует документировать для последующего использования при разработке модели классификации. 