from clearml import PipelineController, Task
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Функция для загрузки данных
def download_data():
    # грузим датасет "california housing prices" — почти классика
    # хотел делать на Бостоне как на парах по эконометрике, но Бостон удалён по этическим причинам(((
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

# Функция для предобработки данных
def preprocess_data(X, y):
    # делим данные на train и test — как пиццу на две неравные части
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Функция для обучения моделей и логирования метрик в ClearML
def train_models(X_train, X_test, y_train, y_test):
    # создаём список для хранения метрик моделей
    metrics_list = []

    # --- Полиномиальная регрессия ---
    # обучаем модель — надеемся, что она не переобучится, как junior на первом проекте
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X_train, y_train)

    # предсказания и метрики
    poly_predictions = poly_model.predict(X_test)
    poly_mse = mean_squared_error(y_test, poly_predictions)
    poly_rmse = np.sqrt(poly_mse)
    poly_mae = mean_absolute_error(y_test, poly_predictions)
    poly_r2 = r2_score(y_test, poly_predictions)

    # логируем метрики полиномиальной регрессии
    task = Task.current_task()
    logger = task.get_logger()
    logger.report_scalar(title="Metrics", series="Polynomial Regression RMSE", value=poly_rmse, iteration=0)
    logger.report_scalar(title="Metrics", series="Polynomial Regression MAE", value=poly_mae, iteration=0)
    logger.report_scalar(title="Metrics", series="Polynomial Regression R2", value=poly_r2, iteration=0)

    # график распределения ошибок для полиномиальной регрессии
    poly_errors = y_test - poly_predictions
    plt.figure(figsize=(8, 6))
    sns.histplot(poly_errors, kde=True, color='purple')
    plt.title("Распределение ошибок полиномиальной регрессии")
    plt.xlabel("Ошибки")
    plt.ylabel("Частота")
    plt.savefig("clearml_poly_error_distribution.png")  # Сохраняем локально
    plt.close()
    logger.report_media(title="Error Distribution", series="Polynomial Regression", local_path="clearml_poly_error_distribution.png")  # Загружаем в ClearML

    # сохраняем метрики полиномиальной регрессии
    metrics_list.append({
        "model": "polynomial_regression",
        "rmse": poly_rmse,
        "mae": poly_mae,
        "r2": poly_r2
    })

    # --- Градиентный бустинг (XGBoost) ---
    # обучаем модель — надеемся, что она не переобучится, как junior на первом проекте
    gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    gb_model.fit(X_train, y_train)

    # предсказания и метрики
    gb_predictions = gb_model.predict(X_test)
    gb_mse = mean_squared_error(y_test, gb_predictions)
    gb_rmse = np.sqrt(gb_mse)
    gb_mae = mean_absolute_error(y_test, gb_predictions)
    gb_r2 = r2_score(y_test, gb_predictions)

    # логируем метрики градиентного бустинга
    logger.report_scalar(title="Metrics", series="Gradient Boosting RMSE", value=gb_rmse, iteration=0)
    logger.report_scalar(title="Metrics", series="Gradient Boosting MAE", value=gb_mae, iteration=0)
    logger.report_scalar(title="Metrics", series="Gradient Boosting R2", value=gb_r2, iteration=0)

    # график распределения ошибок для градиентного бустинга
    gb_errors = y_test - gb_predictions
    plt.figure(figsize=(8, 6))
    sns.histplot(gb_errors, kde=True, color='orange')
    plt.title("Распределение ошибок градиентного бустинга")
    plt.xlabel("Ошибки")
    plt.ylabel("Частота")
    plt.savefig("clearml_gb_error_distribution.png")  # Сохраняем локально
    plt.close()
    logger.report_media(title="Error Distribution", series="Gradient Boosting", local_path="clearml_gb_error_distribution.png")  # Загружаем в ClearML

    # сохраняем метрики градиентного бустинга
    metrics_list.append({
        "model": "gradient_boosting",
        "rmse": gb_rmse,
        "mae": gb_mae,
        "r2": gb_r2
    })

    # --- Сравнение моделей ---
    # график сравнения метрик
    metrics_df = pd.DataFrame(metrics_list)
    plt.figure(figsize=(10, 6))
    metrics_df.set_index("model").plot(kind="bar", y=["rmse", "mae", "r2"], subplots=True, layout=(1, 3), figsize=(15, 5))
    plt.suptitle("Сравнение моделей по метрикам")
    plt.savefig("clearml_model_comparison.png")  # Сохраняем локально
    plt.close()
    logger.report_media(title="Model Comparison", series="Metrics", local_path="clearml_model_comparison.png")  # Загружаем в ClearML

    # визуализация предсказаний
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, poly_predictions, alpha=0.6, label="Polynomial Regression", color='purple')
    plt.scatter(y_test, gb_predictions, alpha=0.6, label="Gradient Boosting", color='orange')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title("Actual vs Predicted California Housing Prices")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.legend()
    plt.savefig("clearml_prediction_plot.png")  # Сохраняем локально
    plt.close()
    logger.report_media(title="Predictions", series="Actual vs Predicted", local_path="clearml_prediction_plot.png")  # Загружаем в ClearML

    return metrics_list

# Настройка пайплайна ClearML
pipe = PipelineController(
    project="California Housing Prices",
    name="ML Pipeline",
    version="0.1",
    add_pipeline_tags=False
)

# Добавление шагов в пайплайн
pipe.add_function_step(
    name="download_data",
    function=download_data,
    function_return=["X", "y"],
    cache_executed_step=False,
)

pipe.add_function_step(
    name="preprocess_data",
    function=preprocess_data,
    function_kwargs=dict(X="${download_data.X}", y="${download_data.y}"),
    function_return=["X_train", "X_test", "y_train", "y_test"],
    cache_executed_step=False,
)

pipe.add_function_step(
    name="train_models",
    function=train_models,
    function_kwargs=dict(
        X_train="${preprocess_data.X_train}",
        X_test="${preprocess_data.X_test}",
        y_train="${preprocess_data.y_train}",
        y_test="${preprocess_data.y_test}",
    ),
    function_return=["metrics_list"]
)

if __name__ == "__main__":
    pipe.start_locally(run_pipeline_steps_locally=True)