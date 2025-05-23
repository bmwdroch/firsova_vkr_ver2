# Глава 3. Разработка приложения (с фокусом на архитектуру)

## 3.4 Выводы по Главе 3

В рамках третьей главы выпускной квалификационной работы были рассмотрены вопросы проектирования и частичной реализации программного приложения, предназначенного для демонстрации работы разработанной ML-модели классификации клиентов по уровню лояльности. Основной акцент был сделан не на создании полнофункционального коммерческого продукта, а на разработке архитектуры и минимально необходимого интерфейса для эффективной демонстрации результатов исследования.

Ключевые результаты, достигнутые в ходе работы над данной главой, включают:

1.  **Спроектирована архитектура приложения:** Разработано архитектурное решение, основанное на API-first подходе и многокомпонентной структуре. Были определены основные компоненты системы: API-сервис на FastAPI, модуль машинного обучения, пользовательский интерфейс на базе Jinja2-шаблонов, а также потенциальное использование базы данных и хранилища файлов. Подробное описание архитектуры вынесено в техническую документацию (`docs/technical/architecture/application_architecture.md`), а в данной главе представлены ее основные аспекты и структура кода.

2.  **Детализировано проектирование основных функциональных блоков:**
    *   Описаны механизмы приема и первичной обработки данных в API-сервисе, включая поддержку загрузки CSV-файлов и ручного ввода данных с валидацией через Pydantic-схемы.
    *   Проработан процесс взаимодействия с ML-моделью: загрузка основной модели LightGBM и всех необходимых трансформеров (скейлеров, энкодеров, PCA, KMeans), а также последовательное применение этих артефактов к новым данным для получения предсказаний. Особое внимание уделено воспроизводимости шагов предобработки и стратегии обработки признаков, которые не могут быть полностью получены из пользовательского ввода.
    *   Спроектирован минималистичный пользовательский интерфейс, включающий отображение информации о модели, формы для ввода данных, вывод результатов классификации и визуализацию распределения клиентов по сегментам лояльности.

3.  **Разработана стратегия тестирования и оценки эффективности:** Предложен план функционального тестирования для API-сервиса и пользовательского интерфейса, охватывающий ключевые сценарии использования. Сформулированы критерии оценки работоспособности и эффективности демонстрационного приложения, ориентированные на корректность выполнения основных функций, стабильность и наглядность представления результатов.

Таким образом, в главе 3 были заложены основы для создания работающего прототипа системы, способного продемонстрировать практическое применение результатов машинного обучения, полученных в предыдущей главе. Спроектированная архитектура обеспечивает гибкость и потенциал для дальнейшего развития системы, в то время как реализованный минимальный функционал достаточен для достижения демонстрационных целей ВКР. Акцент на правильной интеграции ML-модели и корректной обработке данных на этапе предсказания является важным аспектом, подтверждающим жизнеспособность предложенного подхода. 