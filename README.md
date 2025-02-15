# Проект: Анализ данных и машинное обучение на примере набора данных о пассажирах Титаника

## Описание проекта

Данный проект демонстрирует полный цикл работы с данными: от загрузки и предварительной обработки до построения и оценки моделей машинного обучения. В качестве примера используется набор данных о пассажирах Титаника, где цель — предсказать вероятность выживания пассажиров на основе различных признаков (например, возраст, класс каюты, пол и т.д.).

## Суть проекта

Проект решает задачу бинарной классификации: определить, выжил ли пассажир на Титанике или нет. Для этого используются методы анализа данных, предобработки данных и модели машинного обучения. Мы применяем несколько подходов для улучшения качества моделей, включая подбор гиперпараметров, добавление новых признаков, масштабирование данных и использование ансамблевых методов.

## Основные этапы проекта

### Загрузка и предобработка данных  
- Удаление ненужных столбцов (например, `Name`, `Ticket`, `Cabin`).
- Обработка пропущенных значений:
  - Заполнение медианой для возраста (`Age`).
  - Заполнение модальным значением для порта посадки (`Embarked`).
- Кодирование категориальных признаков:
  - Использование метода `pd.get_dummies` для создания числовых представлений категориальных признаков (например, `Sex`, `Embarked`).
- Добавление новых признаков:
  - `AgeGroup`: Группировка пассажиров по возрастным категориям (ребенок, подросток, взрослый, пожилой).
  - `FamilySize`: Размер семьи пассажира (сумма количества братьев/сестер, родителей и детей).
  - `IsAlone`: Признак, указывающий, путешествовал ли пассажир один.
- Масштабирование признаков с помощью `StandardScaler` для улучшения сходимости моделей.

### Исследовательский анализ данных (EDA)  
- Вывод описательной статистики для понимания распределения данных.
- Построение корреляционной матрицы для выявления зависимостей между признаками.
- Визуализация распределения признаков с помощью гистограмм.

### Обучение моделей машинного обучения  
- **Логистическая регрессия**:
  - Простая модель для базового сравнения.
  - Подбор гиперпараметров (например, `C`, `solver`, `penalty`) с использованием `GridSearchCV`.
- **Случайный лес**:
  - Ансамблевый метод, который строит множество деревьев решений.
  - Подбор гиперпараметров (например, `n_estimators`, `max_depth`, `min_samples_split`) для улучшения точности.

### Улучшение качества моделей  
- **Балансировка классов**:
  - Использование метода `SMOTE` для балансировки данных, так как количество выживших и погибших пассажиров несбалансировано.
- **Масштабирование признаков**:
  - Применение `StandardScaler` для нормализации данных, что особенно важно для логистической регрессии.
- **Добавление новых признаков**:
  - Создание дополнительных признаков (`AgeGroup`, `FamilySize`, `IsAlone`) для повышения информативности данных.
- **Подбор гиперпараметров**:
  - Использование `GridSearchCV` для автоматического поиска лучших параметров моделей.
- **Ансамблевые методы**:
  - Применение случайного леса для улучшения точности предсказаний.

### Оценка моделей  
- Использование метрик точности (`accuracy`), полноты (`recall`), точности (`precision`) и `F1`-меры для оценки производительности моделей.
- Сравнение результатов различных моделей (логистическая регрессия, случайный лес).

## Результаты

После применения всех улучшений и оптимизации моделей были получены следующие результаты:

| Модель | Accuracy | Recall | Precision | F1-score |
|--------|---------|--------|-----------|----------|
| Логистическая регрессия | 0.82 | 0.81 | 0.81 | 0.81 |
| Случайный лес | 0.83 | 0.82 | 0.82 | 0.82 |

## Как использовать проект

### Установка зависимостей  
```bash
pip install -r requirements.txt
```

### Запуск проекта  
```bash
python src/main.py
```

### Ожидаемые выходные данные  
- Корреляционная матрица и гистограммы признаков.
- Точность и отчет классификации для каждой модели.

## Что я применила для улучшения качества моделей?

- **Предобработка данных**:
  - Обработка пропущенных значений и удаление ненужных столбцов.
  - Добавление новых признаков (`AgeGroup`, `FamilySize`, `IsAlone`).
- **Масштабирование данных**:
  - Использование `StandardScaler` для нормализации признаков.
- **Балансировка классов**:
  - Применение метода `SMOTE` для устранения дисбаланса между классами.
- **Подбор гиперпараметров**:
  - Использование `GridSearchCV` для автоматического поиска оптимальных параметров моделей.
- **Ансамблевые методы**:
  - Применение случайного леса для повышения точности предсказаний.
- **Оценка моделей**:
  - Использование метрик точности, полноты, точности и `F1`-меры для сравнения производительности моделей.

## Заключение

Этот проект демонстрирует, как можно улучшить качество моделей машинного обучения с помощью комплексного подхода: от предобработки данных до использования продвинутых алгоритмов. Применение методов балансировки классов, масштабирования данных и ансамблевых методов позволило значительно повысить точность предсказаний.

