# Student At-Risk Prediction System

Predicts the likelihood of a university student dropping out using demographic, financial, and academic performance data. Trained on the UCI Predict Students' Dropout and Academic Success dataset.

## Dataset

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

4,424 students | 36 features | Binary target: Dropout (at-risk) vs Graduate/Enrolled (safe)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Step 1 — Train the model:**
```bash
python train.py
```
Outputs trained model artifacts (`*.pkl`) and evaluation charts (`*.png`).

**Step 2 — Launch the dashboard:**
```bash
streamlit run dashboard.py
```
Opens a browser interface where you can enter a student's details and get an at-risk prediction with confidence score.

## Files

| File | Description |
|---|---|
| `train.py` | ML pipeline: loads UCI data, trains 3 models, evaluates, saves artifacts |
| `dashboard.py` | Streamlit web interface for interactive prediction |
| `uci_data.csv` | Raw dataset (semicolon-separated, UTF-8 BOM) |
| `requirements.txt` | Python dependencies |

## Models

Three classifiers are trained and compared — Logistic Regression, Decision Tree, and Random Forest — all with `class_weight='balanced'` to handle the 32/68 class imbalance. The best model is selected by F1 score on the at-risk class (not overall accuracy). Logistic Regression currently achieves the best result: 83.1% recall, 80.8% F1 on the test set.
