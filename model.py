from __future__ import annotations
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from numpy.typing import ArrayLike
from joblib import dump

from data_utils import load_and_clean_dou, sort_titles_by_seniority

DOU_CSV = "dou_raw.csv"


def train_regression_model(X_train: ArrayLike, y_train: ArrayLike) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def save_regression_model(model: LinearRegression, filename: str = "linear_regression_model.joblib"):
    dump(model, filename)


def evaluate_regression_model(model: LinearRegression, X_test: ArrayLike, y_test: ArrayLike):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return mse


def save_initial_datasets(X: ArrayLike, y: ArrayLike):
    dump(X, "X.joblib")
    dump(y, "y.joblib")


def build_salary_pipeline() -> Pipeline:
    categorical_features = ["title", "category", "language_group"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", LinearRegression()),
    ])


if __name__ == '__main__':
    df = load_and_clean_dou(DOU_CSV)
    print(f"Завантажено та очищено {len(df)} рядків з реального датасету DOU.")

    # Tab 1: стаж -> індекс сивини
    X = df[["experience_years"]].values
    y = df["sivyna_index"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sivyna_model = train_regression_model(X_train, y_train)
    evaluate_regression_model(sivyna_model, X_test, y_test)
    save_regression_model(sivyna_model, "linear_regression_model.joblib")
    save_initial_datasets(X, y)

    # Tab 2: стаж + вік + тайтл + категорія + англ. + мова -> зарплата
    feature_cols = ["experience_years", "age", "title", "category", "english_level", "language_group"]
    X_salary = df[feature_cols]
    y_salary = df["salary"].values

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_salary, y_salary, test_size=0.2, random_state=42
    )

    salary_pipeline = build_salary_pipeline()
    salary_pipeline.fit(Xs_train, ys_train)

    salary_pred = salary_pipeline.predict(Xs_test)
    salary_mse = mean_squared_error(ys_test, salary_pred)
    salary_r2 = r2_score(ys_test, salary_pred)
    print(f"[Зарплата] MSE: {salary_mse:.1f}, R^2: {salary_r2:.3f}")

    dump(salary_pipeline, "salary_model.joblib")

    dropdown_values = {
        "titles": sort_titles_by_seniority(df["title"].unique().tolist()),
        "categories": sorted(df["category"].unique().tolist()),
        "languages": sorted(df["language_group"].unique().tolist()),
    }
    dump(dropdown_values, "dropdown_values.joblib")

    print("Готово: моделі та датасети збережено (*.joblib).")