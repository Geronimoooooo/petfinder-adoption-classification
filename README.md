# PetFinder Adoption — бинарная классификация / binary classification

> **RU.** Бинарная классификация «быстро / не быстро усыновят» для
> животных из приютов Малайзии (Kaggle *PetFinder.my Adoption
> Prediction*).
>
> **EN.** Binary classification — will a Malaysian shelter animal be
> adopted quickly or not (Kaggle *PetFinder.my Adoption Prediction*).

---

## Описание / Overview

**RU.** Исходная Kaggle-задача 5-классовая: `AdoptionSpeed` принимает
значения 0–4 (скорость усыновления). Для учебного сравнения моделей
задача сводится к бинарной: 0–2 («быстро» — в течение ≈месяца) → `0`;
3–4 (медленно или никогда) → `1`. Реализован полный цикл: очистка и
отбор признаков, две стратегии масштабирования (разные требования у
`BernoulliNB`/`KNN` и `MultinomialNB`), one-hot для категориальных,
подбор гиперпараметров `GridSearchCV` по `ROC-AUC`, сравнение трёх
моделей на валидации.

**EN.** The original Kaggle task has 5 classes for `AdoptionSpeed`
(0–4). To compare models clearly we collapse it to binary: 0–2 (“fast”,
within ≈a month) → `0`; 3–4 (slow or never) → `1`. The notebook covers
the full workflow — feature cleaning and selection, two scaling
strategies (to meet the distinct requirements of `BernoulliNB`/`KNN`
vs. `MultinomialNB`), one-hot encoding for categorical features,
hyperparameter tuning with `GridSearchCV` on `ROC-AUC`, and a
head-to-head comparison on the validation set.

## Датасет / Dataset

- **Источник / Source:**
  <https://www.kaggle.com/c/petfinder-adoption-prediction>
- Зеркало `train.csv` (Google Sheets export) подключается
  автоматически в первой секции ноутбука.
- **Целевая переменная / Target:** `AdoptionSpeed_binary` (0 — быстро,
  1 — медленно / не усыновили).

## Стек / Stack

- Python 3.11
- `numpy`, `pandas`
- `scikit-learn` (`RobustScaler`, `MinMaxScaler`, `GridSearchCV`,
  `BernoulliNB`, `MultinomialNB`, `KNeighborsClassifier`)
- `requests`

## Структура / Structure

```
petfinder-adoption-classification/
├── README.md
└── petfinder_adoption_classification.ipynb
```

Логические разделы / notebook sections:

1. Импорты / Imports
2. Загрузка данных / Data loading
3. Целевая переменная / Target engineering
4. Отбор признаков / Feature selection
5. Разделение num/cat / Numeric vs. categorical split
6. Два варианта масштабирования / Two scaling strategies
   (RobustScaler для BernoulliNB/KNN; MinMaxScaler + квантильный
   клиппинг для MultinomialNB)
7. One-hot для категориальных / One-hot for categorical
8. Train/val split
9. GridSearchCV по трём моделям
10. Функция оценки / Evaluation helper
11. Обучение финальных моделей и сравнение / Training and comparison
12. Выводы / Conclusions

## Результаты / Results

**RU.**

- `BernoulliNB` показал лучший `ROC-AUC ≈ 0.65` на валидации — его
  бинарная природа отлично согласуется с one-hot признаками.
- `MultinomialNB` чуть отстаёт, но даёт достойный бейзлайн при
  неотрицательных нормированных входах.
- `KNN` (`n_neighbors=21`) хуже из-за высокой размерности признакового
  пространства после one-hot.
- `GaussianNB` не рассматривается: распределение one-hot признаков не
  гауссово.

**EN.**

- `BernoulliNB` leads with `ROC-AUC ≈ 0.65` on the validation set —
  its binary nature matches one-hot features nicely.
- `MultinomialNB` trails slightly but remains a solid baseline with
  properly normalised non-negative inputs.
- `KNN` (`n_neighbors=21`) underperforms due to the high
  dimensionality of the one-hot feature space.
- `GaussianNB` is excluded: one-hot feature distributions are not
  Gaussian.

## Как запустить / How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install numpy pandas scikit-learn requests jupyter

jupyter notebook petfinder_adoption_classification.ipynb
```

**RU.** `train.csv` (~5 МБ) скачивается автоматически при первом
запуске ноутбука.

**EN.** `train.csv` (~5 MB) is downloaded automatically on the first
run of the notebook.
