import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# грузим датасет "california housing prices" — почти классика
# хотел делать на Бостоне как на парах по эконометрике, но Бостон удалён по этическим причинам(((
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# делим данные на train и test — как пиццу на две неравные части
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# настраиваем MLflow — тут будет магия
mlflow.set_experiment("california_housing_experiment")

# параметры модели — играемся, как с настройками в конфиге
n_estimators = 200
max_depth = 10

# создаём список для хранения метрик моделей
metrics_list = []

with mlflow.start_run():
    # --- Линейная регрессия ---
    # обучаем модель — надеемся, что она не переобучится, как junior на первом проекте
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # предсказания и метрики
    linear_predictions = linear_model.predict(X_test)
    linear_mse = mean_squared_error(y_test, linear_predictions)
    linear_rmse = np.sqrt(linear_mse)
    linear_mae = mean_absolute_error(y_test, linear_predictions)
    linear_r2 = r2_score(y_test, linear_predictions)

    # логируем метрики линейной регрессии
    mlflow.log_metric("linear_rmse", linear_rmse)
    mlflow.log_metric("linear_mae", linear_mae)
    mlflow.log_metric("linear_r2", linear_r2)

    # график распределения ошибок для линейной регрессии
    linear_errors = y_test - linear_predictions
    plt.figure(figsize=(8, 6))
    sns.histplot(linear_errors, kde=True, color='blue')
    plt.title("Распределение ошибок линейной регрессии")
    plt.xlabel("Ошибки")
    plt.ylabel("Частота")
    plt.savefig("mlflow_linear_error_distribution.png")
    plt.close()

    # сохраняем метрики линейной регрессии
    metrics_list.append({
        "model": "linear_regression",
        "rmse": linear_rmse,
        "mae": linear_mae,
        "r2": linear_r2
    })

    # --- Случайный лес ---
    # обучаем модель — надеемся, что она не переобучится, как junior на первом проекте
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(X_train, y_train)

    # предсказания и метрики
    rf_predictions = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(rf_mse)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)

    # логируем метрики случайного леса
    mlflow.log_metric("rf_rmse", rf_rmse)
    mlflow.log_metric("rf_mae", rf_mae)
    mlflow.log_metric("rf_r2", rf_r2)

    # график распределения ошибок для случайного леса
    rf_errors = y_test - rf_predictions
    plt.figure(figsize=(8, 6))
    sns.histplot(rf_errors, kde=True, color='green')
    plt.title("Распределение ошибок случайного леса")
    plt.xlabel("Ошибки")
    plt.ylabel("Частота")
    plt.savefig("mlflow_rf_error_distribution.png")
    plt.close()

    # сохраняем метрики случайного леса
    metrics_list.append({
        "model": "random_forest",
        "rmse": rf_rmse,
        "mae": rf_mae,
        "r2": rf_r2
    })

    # --- Сравнение моделей ---
    # график сравнения метрик
    metrics_df = pd.DataFrame(metrics_list)
    plt.figure(figsize=(10, 6))
    metrics_df.set_index("model").plot(kind="bar", y=["rmse", "mae", "r2"], subplots=True, layout=(1, 3), figsize=(15, 5))
    plt.suptitle("Сравнение моделей по метрикам")
    plt.savefig("mlflow_model_comparison.png")
    plt.close()

    # логируем модели с примерами и подписями
    input_example = X_test.iloc[:5]
    signature_linear = infer_signature(X_train, linear_model.predict(X_train))
    signature_rf = infer_signature(X_train, rf_model.predict(X_train))

    mlflow.sklearn.log_model(linear_model, "linear_regression", input_example=input_example, signature=signature_linear)
    mlflow.sklearn.log_model(rf_model, "random_forest_regressor", input_example=input_example, signature=signature_rf)

    # сохраняем метрики в файл — на всякий случай, вдруг MLflow сломается
    metrics_df.to_csv("mlflow_metrics.csv", index=False)

    # визуализация предсказаний
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, linear_predictions, alpha=0.6, label="Linear Regression")
    plt.scatter(y_test, rf_predictions, alpha=0.6, label="Random Forest")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title("Actual vs Predicted California Housing Prices")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.legend()
    plt.savefig("mlflow_prediction_plot.png")