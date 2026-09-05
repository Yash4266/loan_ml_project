# Loan Approval ML Practice Project

This project follows the basic ML flow:

Dataset (`loan.csv`) -> Feature Engineering -> Data Preprocessing -> Train/Test Split -> ML Model -> Prediction (Accuracy) -> F1-score / Precision

IMPORTANT: `dataset/loan.csv` is a SYNTHETIC practice dataset created for learning. It is not real bank/customer data and must not be used for real loan decisions.

## Project structure

```
loan_ml_project/
  dataset/loan.csv
  model.py
  requirements.txt
  README.md
```

GitHub -> csv file -> python file (all features in code)

## Run

```bash
cd loan_ml_project
pip install -r requirements.txt
python model.py
```

## Steps in `model.py`

1. **Dataset** — load `dataset/loan.csv`
2. **Feature engineering** — keep all original features and add `age_group`, `income_level`, `credit_level`, `loan_to_income_ratio`
3. **Data preprocessing** — fill missing values (median), then scale features
4. **Train/Test split** — 80% train, 20% test (after the first 3 steps)
5. **ML model** — Logistic Regression
6. **Prediction (Accuracy)**
7. **F1-score / Precision**

## Target

`target`:

- 0 = loan rejected
- 1 = loan approved

## Features

Original: age, gender, income, loan_amount, loan_term, credit_score, employment_years, education, married, dependents, existing_loans, property_area, self_employed

Engineered: age_group, income_level, credit_level, loan_to_income_ratio
