from __future__ import annotations
import re
import numpy as np
import pandas as pd

RAW_CSV_URL = "https://raw.githubusercontent.com/devua/csv/refs/heads/master/salaries/2025_dec_raw.csv"

SALARY_COL = "ЗАРПЛАТА / СУМАРНИЙ ДОХІД в IT у $$$ за місяць, лише ставка \nЧИСТИМИ - після сплати податків"
BONUS_COL = "Всі бонуси (на місяць)"
STAGE_COL = "Загальний стаж роботи за нинішньою ІТ-спеціальністю"
TITLE_COL = "Тайтл"
CATEGORY_COL = "Категорії"
ENGLISH_COL = "Знання англійської мови"
LANG_COL = "Основна мова програмування"
AGE_COL = "Вік"

ENGLISH_ORDER = {
    "Не знаю взагалі": 0,
    "Elementary": 0,
    "Pre-Intermediate": 1,
    "Intermediate": 2,
    "Upper-Intermediate": 3,
    "Advanced": 4,
}

TITLE_RENAME = {
    "Немає тайтлу": "Junior",
    "Lead": "Team Lead",
    "Team Lead (for SE & QA: Lead + Team Lead)": "Team Lead",
    "Technical Lead": "Tech Lead",
    "Manager": "Head",
    "Consultant / External Expert": "Principal",
    "CEO / C-level (Chief) / Director / VP": "C-level",
}

TITLE_SENIORITY_ORDER = [
    "Intern/Trainee", "Junior", "Middle", "Senior", "Staff",
    "Team Lead", "Tech Lead", "Architect", "Principal", "Head", "C-level",
]


def sort_titles_by_seniority(titles) -> list:
    known = [t for t in TITLE_SENIORITY_ORDER if t in titles]
    unknown = sorted(t for t in titles if t not in TITLE_SENIORITY_ORDER)
    return known + unknown


TOP_LANGUAGES = [
    "TypeScript", "Python", "JavaScript", "C#  NET", "Java", "PHP", "SQL",
    "C++", "Kotlin", "Swift", "Go", "Bash  Shell", "Ruby", "Dart",
]

CATEGORY_GROUPS = {
    "Support (other)": "Support",
    "Technical Support": "Support",
    "Customer Support": "Support",
    "Other": "Other",
    "Other Engineering": "Other",
    "Other Management": "Other",
    "Tech Leadership": "Leadership",
    "General Leadership": "Leadership",
}


def group_category(value: str) -> str:
    if pd.isna(value):
        return "Інше"
    return CATEGORY_GROUPS.get(value, value)


_EXPERIENCE_MAP = {
    "менше як 3 місяці": 0.15,
    "3 місяці": 0.25,
    "пів року": 0.5,
    "півтора роки": 1.5,
    "1 рік": 1,
    "2 роки": 2,
    "3 роки": 3,
    "4 роки": 4,
    "5 років": 5,
    "6 років": 6,
    "7 років": 7,
    "8 років": 8,
    "9 років": 9,
    "10 років": 10,
    "11 років": 11,
    "12 років": 12,
    "13 років": 13,
    "14 років": 14,
    "15-20 років": 17.5,
    "понад 20 років": 22,
}


def parse_experience(value: str) -> float:
    # переводить текстовий стаж типу "15-20 років" чи "Пів року" в число років
    if pd.isna(value):
        return np.nan

    key = str(value).strip().lower()
    if key in _EXPERIENCE_MAP:
        return _EXPERIENCE_MAP[key]

    range_match = re.search(r"(\d+)\s*-\s*(\d+)", key)
    if range_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        return (lo + hi) / 2

    number_match = re.search(r"(\d+)", key)
    if number_match:
        n = int(number_match.group(1))
        if "місяц" in key:
            return round(n / 12, 2)
        return float(n)

    return np.nan


def compute_sivyna_index(experience_years: pd.Series, age: pd.Series) -> pd.Series:
    raw = experience_years * 3.2 + (age - 24).clip(lower=0) * 1.4
    return raw.clip(lower=0, upper=100)


def coffee_cups_from_sivyna(sivyna_index: pd.Series | float) -> pd.Series | float:
    return 1 + (sivyna_index / 100.0) * 6


def group_language(value) -> str:
    if pd.isna(value):
        return "Не програмує"
    return value if value in TOP_LANGUAGES else "Інша"


def load_and_clean_dou(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["experience_years"] = df[STAGE_COL].apply(parse_experience)
    df["age"] = df[AGE_COL].astype(float)
    df["salary"] = df[SALARY_COL].astype(float)
    df["title"] = df[TITLE_COL].replace(TITLE_RENAME)
    df["category"] = df[CATEGORY_COL].apply(group_category)
    df["english_level"] = df[ENGLISH_COL].map(ENGLISH_ORDER).fillna(0).astype(int)
    df["language_group"] = df[LANG_COL].apply(group_language)

    df["sivyna_index"] = compute_sivyna_index(df["experience_years"], df["age"])
    df["coffee_cups"] = coffee_cups_from_sivyna(df["sivyna_index"])

    df = df.dropna(subset=["experience_years", "age", "salary"])

    salary_cap = df["salary"].quantile(0.99)
    df = df[df["salary"] <= salary_cap]

    keep_cols = [
        "experience_years", "age", "sivyna_index", "coffee_cups", "salary",
        "title", "category", "english_level", "language_group",
    ]
    return df[keep_cols].reset_index(drop=True)