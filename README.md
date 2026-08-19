# Heart Disease ML Practice Project

This project follows the basic ML flow:

Dataset (`heart.csv`) -> Feature Engineering -> Data Preprocessing -> Train/Test Split -> ML Model -> Prediction (Accuracy) -> F1-score / Precision

IMPORTANT: `dataset/heart.csv` is a SYNTHETIC practice dataset created for learning. It is not real patient/medical data and must not be used for medical decisions.

## Project structure

```
heart_ml_project/
  dataset/heart.csv
  model.py
  requirements.txt
  README.md
```

GitHub -> csv file -> python file (all features in code)

## Run

```bash
cd heart_ml_project
pip install -r requirements.txt
python model.py
```

## Steps in `model.py`

1. **Dataset** — load `dataset/heart.csv`
2. **Feature engineering** — keep all original features and add `age_group`, `chol_level`, `bp_level`, `max_hr_ratio`
3. **Data preprocessing** — fill missing values (median), then scale features
4. **Train/Test split** — 80% train, 20% test (after the first 3 steps)
5. **ML model** — Logistic Regression
6. **Prediction (Accuracy)**
7. **F1-score / Precision**

## Target

`target`:

- 0 = lower-risk class in this synthetic practice dataset
- 1 = higher-risk class in this synthetic practice dataset

## Features

Original: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

Engineered: age_group, chol_level, bp_level, max_hr_ratio
