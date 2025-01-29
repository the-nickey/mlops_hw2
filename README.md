## MLOps - ДЗ 2
В этой репе второе домашнее задание по MLOps
Сделал Павел Очкин
Я делал только части 2 и 3 (DVC не осилил, увы)


# *Что в репе*?

Структурирование репозитория в этом домашнем задании не оценивается, поэтому всё немного в кучу))0)

Здесь есть:
- скрипт с MLFlow - mlflow_experiment.py
- пайплайн ClearML - clearml_pipeline.py
- метрики моделей - metrics.csv
- графики, которые выдали эксперименты - всё, что заканчивается на .png
- сервисные файлы в папках


# *Что у меня получилось?*

- В первую очередь - разобраться с обеими технологиями и прогнать локально
Это было существенно сложнее, чем я ожидал - на работе мне недавно дали Mac, и это мой основной компьютер для работы, учёбы и жизни.
Для того, чтобы ClearML завёлся, пришлось менять LibreSSL на OpenSSL и переустанавливать Python.
Я обычно не делаю виртуальное окружение, но для этого проекта сделал - оно легло в папку mlflow_env.


Как я работал с MLFlow:
<img width="1669" alt="Снимок экрана 2025-01-30 в 00 24 30" src="https://github.com/user-attachments/assets/82a9e280-b56b-42e7-9a58-164e1abb9b4c" />
<img width="1671" alt="Снимок экрана 2025-01-30 в 00 24 41" src="https://github.com/user-attachments/assets/4fb7ca6e-963e-4ca4-a1c6-6f6d1d6e1d90" />
<img width="1670" alt="Снимок экрана 2025-01-30 в 00 24 56" src="https://github.com/user-attachments/assets/a24d1415-78fb-434c-a946-04ae8765fcfc" />
<img width="1670" alt="Снимок экрана 2025-01-30 в 00 25 07" src="https://github.com/user-attachments/assets/9be06dd2-d3b5-4939-bf34-3a2bc537a45e" />
<img width="1666" alt="Снимок экрана 2025-01-30 в 00 58 54" src="https://github.com/user-attachments/assets/42181511-300f-4f8a-ade6-106969b5a327" />


Как поработалось с ClearML:
<img width="1671" alt="Снимок экрана 2025-01-30 в 00 59 10" src="https://github.com/user-attachments/assets/9fab5e8e-feb7-4698-b18d-c6f5ca18d9bb" />
<img width="1670" alt="Снимок экрана 2025-01-30 в 00 59 27" src="https://github.com/user-attachments/assets/01d5fced-8932-4789-bb6e-17e5dd960112" />


# *Что получилось в экспериментах?*
Я знаю и "классический" ML, и "продвинутые" методы - на работе часто приходится использовать catboost.
Поэтому в экспериментах я не просто менял параметры моделей, а сопоставлял "классические" методы с "продвинутыми":
- в MLFlow я реализовал классическую линейную регрессию и регрессию через случайный лес
- в ClearML сделал полиномиальную регрессию и градиентный бустинг в XGBoost

## Результаты экспериментов

# Часть 2 - MLFlow

График предсказаний линейной регрессии и случайного леса

![mlflow_prediction_plot](https://github.com/user-attachments/assets/a19078c7-f969-4dbb-9570-bfd194d3b227)

Распределение ошибок линейной модели

![mlflow_linear_error_distribution](https://github.com/user-attachments/assets/42e62176-a1d3-43d7-a1a6-c2085d1a6be7)

Распределение ошибок случайного леса

![mlflow_rf_error_distribution](https://github.com/user-attachments/assets/a7513eb9-668b-4f53-91d7-c3f472b5065f)

Сопоставление метрик моделей

![mlflow_model_comparison](https://github.com/user-attachments/assets/f21c8715-7ec2-4f8b-9764-973dfdf221f7)



# Часть 3 - ClearML

График предсказаний полиномиальной регрессии и градиентного бустинга

![clearml_prediction_plot](https://github.com/user-attachments/assets/e04177ce-ad7e-4875-b97c-33f1033cde53)

Распределение ошибок полиномиальной модели

![clearml_poly_error_distribution](https://github.com/user-attachments/assets/de61c938-eb1d-4d4e-8019-74ab11c0be66)

Распределение ошибок градиентного бустинга

![clearml_gb_error_distribution](https://github.com/user-attachments/assets/0f054483-f644-44aa-8614-884154d4f9da)

Сопоставление метрик моделей

![clearml_model_comparison](https://github.com/user-attachments/assets/cdb1eda1-161d-4ef6-88f7-01906c561970)


"Продвинутый" ML выиграл - лучшие результаты дали XGBoost (R^2 = 0,83) и randomforest (R^2 = 0,78)
