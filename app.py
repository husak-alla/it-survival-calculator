from __future__ import annotations
import math
import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from data_utils import ENGLISH_ORDER

BG_COLOR = "#121212"
PANEL_COLOR = "#332419"
PRIMARY_COLOR = "#D4A373"
TEXT_COLOR = "#F5EBE0"
MUTED_COLOR = "#8C7B6B"


def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> ArrayLike:
    model = load(filename)
    return model.predict(X)


def _index_of_closest(X: ArrayLike, k: float) -> int:
    X = np.asarray(X)
    return (np.abs(X - k)).argmin()


def visualize_sivyna_difference(input_feature: float, prediction: ArrayLike):
    X = load("X.joblib")
    y = load("y.joblib")

    closest_idx = _index_of_closest(X, input_feature)
    actual_target = y[closest_idx]
    difference = actual_target - prediction[0]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.scatter(X, y, color=MUTED_COLOR, alpha=0.35, label="Датасет DOU", s=15)
    ax.scatter(input_feature, actual_target, color=TEXT_COLOR, s=90, zorder=3,
               edgecolor=BG_COLOR, linewidth=1, label="Найближчий реальний респондент")
    ax.scatter(input_feature, prediction[0], color=PRIMARY_COLOR, s=90, zorder=3,
               edgecolor=BG_COLOR, linewidth=1, label="Прогноз моделі")
    ax.plot([input_feature, input_feature], [actual_target, prediction[0]],
            color=TEXT_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    ax.annotate(
        f"Різниця = {difference:.1f}",
        xy=(input_feature, (actual_target + prediction[0]) / 2),
        xytext=(10, 0), textcoords="offset points", color=TEXT_COLOR, fontsize=9,
    )

    ax.legend(facecolor=PANEL_COLOR, edgecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
    ax.set_title("Індекс сивини: прогноз vs реальні дані DOU", color=TEXT_COLOR, fontsize=12)
    ax.set_xlabel("Стаж в ІТ (роки)", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Індекс сивини (%)", color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(MUTED_COLOR)
    ax.grid(True, alpha=0.15, color=MUTED_COLOR)

    st.pyplot(fig)


def tab_sivyna_detector():
    st.subheader("Індекс сивини")
    st.caption("ІТ-стаж має побічний ефект. Дізнайся свій рівень сивини.")

    experience = st.slider("Ваш стаж в ІТ (років)", 0.0, 25.0, 3.0, step=0.5, key="sivyna_slider")

    if st.button("Порахувати рівень сивини", key="sivyna_button"):
        prediction = load_and_predict([[experience]])
        sivyna_pct = float(np.clip(prediction[0], 0, 100))

        st.metric("Прогнозований індекс сивини", f"{sivyna_pct:.1f} %")

        visualize_sivyna_difference(experience, prediction)


COFFEE_PRICE = 3.0
THERAPY_PRICE = 50.0
TEA_PRICE = 1.0
SAVINGS_PRICE = 1.0

COFFEE_PRODUCTIVITY = 0.5
THERAPY_PRODUCTIVITY = 3.0
TEA_PRODUCTIVITY = 0.15
SAVINGS_PRODUCTIVITY = 0.05

COFFEE_MAX_PER_MONTH = 90
THERAPY_MAX_PER_MONTH = 4
TEA_MAX_PER_MONTH = 60
SAVINGS_MAX_PER_MONTH = 1_000_000


def predict_salary(experience_years, age, title, category, english_level, language_group) -> float:
    pipeline = load("salary_model.joblib")
    row = pd.DataFrame([{
        "experience_years": experience_years,
        "age": age,
        "title": title,
        "category": category,
        "english_level": english_level,
        "language_group": language_group,
    }])
    prediction = pipeline.predict(row)
    return max(float(prediction[0]), 0.0)


def solve_survival_budget(
    monthly_budget: float,
    coffee_weight: float = 1.0,
    therapy_weight: float = 1.0,
    tea_weight: float = 1.0,
) -> dict:
    # заощадження - четверта змінна з найнижчим пріоритетом,
    # поглинає залишок бюджету, щоб він завжди був розписаний повністю
    c = [
        -COFFEE_PRODUCTIVITY * coffee_weight,
        -THERAPY_PRODUCTIVITY * therapy_weight,
        -TEA_PRODUCTIVITY * tea_weight,
        -SAVINGS_PRODUCTIVITY,
    ]
    A_ub = [[COFFEE_PRICE, THERAPY_PRICE, TEA_PRICE, SAVINGS_PRICE]]
    b_ub = [monthly_budget]
    bounds = [
        (0, COFFEE_MAX_PER_MONTH),
        (0, THERAPY_MAX_PER_MONTH),
        (0, TEA_MAX_PER_MONTH),
        (0, SAVINGS_MAX_PER_MONTH),
    ]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    coffee_raw, therapy_raw, tea_raw, savings_raw = result.x

    coffee_month = math.floor(coffee_raw)
    therapy_month = math.floor(therapy_raw)
    tea_month = math.floor(tea_raw)

    spent = coffee_month * COFFEE_PRICE + therapy_month * THERAPY_PRICE + tea_month * TEA_PRICE
    savings_month = monthly_budget - spent

    return {
        "coffee_cups_per_month": coffee_month,
        "therapy_sessions_per_month": therapy_month,
        "tea_cups_per_month": tea_month,
        "savings_per_month": savings_month,
        "spent": spent,
        "leftover": monthly_budget - spent - savings_month,
        "productivity_score": -result.fun if result.success else 0.0,
        "success": result.success,
    }


def tab_coffee_optimizer():
    st.subheader("Оптимізація виживання")
    st.caption("Кава, чай чи терапія — обираємо оптимальний рецепт виживання в ІТ за твоєю зарплатою.")

    dropdowns = load("dropdown_values.joblib")

    col1, col2 = st.columns(2)
    with col1:
        experience = st.slider("Стаж в ІТ (років)", 0.0, 25.0, 3.0, step=0.5, key="opt_experience")
        age = st.slider("Вік", 18, 65, 27, key="opt_age")
        title = st.selectbox("Тайтл", dropdowns["titles"], key="opt_title")
    with col2:
        category = st.selectbox("Категорія", dropdowns["categories"], key="opt_category")
        english_label = st.selectbox("Англійська", list(ENGLISH_ORDER.keys())[1:], index=2, key="opt_english")
        language = st.selectbox("Мова програмування", dropdowns["languages"], key="opt_language")

    budget_pct = st.slider(
        "Скільки % прогнозованої зарплати готові витрачати на 'виживання'?",
        5, 40, 15, key="opt_budget_pct"
    )

    st.write("Особисті пріоритети (0.5 — не цікавить, 2.0 — дуже важливо):")
    w1, w2, w3 = st.columns(3)
    with w1:
        coffee_weight = st.slider("Кава", 0.5, 2.0, 1.0, step=0.1, key="opt_coffee_weight")
    with w2:
        therapy_weight = st.slider("Терапія", 0.5, 2.0, 1.0, step=0.1, key="opt_therapy_weight")
    with w3:
        tea_weight = st.slider("Чай", 0.5, 2.0, 1.0, step=0.1, key="opt_tea_weight")

    if st.button("Розрахувати оптимальний бюджет", key="opt_button"):
        english_level = ENGLISH_ORDER[english_label]
        salary = predict_salary(experience, age, title, category, english_level, language)
        monthly_budget = salary * budget_pct / 100

        st.metric("Прогнозована зарплата", f"${salary:,.0f}/міс")
        st.write(f"Бюджет на виживання ({budget_pct}%): **${monthly_budget:,.0f}/міс**")

        result = solve_survival_budget(monthly_budget, coffee_weight, therapy_weight, tea_weight)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Кави на місяць", f"{result['coffee_cups_per_month']}")
        c2.metric("Терапія на місяць", f"{result['therapy_sessions_per_month']}")
        c3.metric("Чаю на місяць", f"{result['tea_cups_per_month']}")
        c4.metric("Відкладено", f"${result['savings_per_month']:.0f}")

        st.markdown(
            f"""
            <div style="background-color:{PANEL_COLOR}; border:1px solid {PRIMARY_COLOR};
                        border-radius:8px; padding:16px 20px; color:{TEXT_COLOR}; line-height:1.5;">
            <strong style="color:{PRIMARY_COLOR};">Вердикт:</strong>
            При стажі {experience:g} р. вам оптимально пити
            {result['coffee_cups_per_month']} чашки кави на місяць і ходити до психотерапевта
            {result['therapy_sessions_per_month']} раз(и) на місяць.
            Ще ${result['savings_per_month']:.0f}/міс можна відкласти на відпустку.
            Очі майже не сіпаються.
            </div>
            """,
            unsafe_allow_html=True,
        )


def create_streamlit_app():
    st.title("Калькулятор виживання в ІТ")

    tab1, tab2 = st.tabs(["Індекс сивини", "Оптимізація виживання"])
    with tab1:
        tab_sivyna_detector()
    with tab2:
        tab_coffee_optimizer()


if __name__ == '__main__':
    create_streamlit_app()