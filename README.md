# Heart Disease Classification – ML Assignment 2

## a. Problem Statement
Predict the presence of heart disease (binary: 0 = no, 1 = yes) from 13 clinical attributes.

## b. Dataset Description
Source: UCI / Kaggle Heart Disease dataset (heart.csv).
Rows: ___ (after removing duplicates). Features: 13 (age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal). Target: `target` (0/1).

## c. GitHub Repository Link
<paste link>

## d. Models Used
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

(Fill from `python train.py` output / model/metrics.csv)

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| Overall winner | |

## How to run
```
pip install -r requirements.txt
python train.py        # needs heart.csv in this folder
streamlit run app.py
```
