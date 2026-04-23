# ============================================================
#  STUDENT AT-RISK PREDICTION SYSTEM
#  Problem Statement: Predict academic performance to identify
#  at-risk students early (before they drop out).
#
#  Dataset: Predict Students' Dropout and Academic Success
#  Source:  UCI Machine Learning Repository
#  Citation:
#    Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L.
#    (2021). Predict Students' Dropout and Academic Success
#    [Dataset]. UCI Machine Learning Repository.
#    https://doi.org/10.24432/C5MC89
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from matplotlib.patches import Patch

print("=" * 62)
print("  STUDENT AT-RISK PREDICTION — UCI Real Dataset")
print("  Citation: Realinho et al. (2021), UCI ML Repository")
print("  DOI: https://doi.org/10.24432/C5MC89")
print("=" * 62)
print()

# ============================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================
df = pd.read_csv('uci_data.csv', sep=';', encoding='utf-8-sig')
df.columns = [c.strip() for c in df.columns]   # remove any stray \t from col names

# Binary target: Dropout = at-risk (1), Graduate/Enrolled = safe (0)
df['at_risk'] = (df['Target'] == 'Dropout').astype(int)

FEATURE_COLS = [c for c in df.columns if c not in ('Target', 'at_risk')]
X = df[FEATURE_COLS]
y = df['at_risk']

print(f"Dataset loaded  : {df.shape[0]} rows x {len(FEATURE_COLS)} features")
print(f"At-Risk (1)     : {(y == 1).sum()}  ({(y == 1).mean()*100:.1f}%)")
print(f"Not At-Risk (0) : {(y == 0).sum()}  ({(y == 0).mean()*100:.1f}%)")
print()

# ============================================================
# STEP 2: TRAIN/TEST SPLIT + SCALING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train size : {X_train_sc.shape[0]} students")
print(f"Test size  : {X_test_sc.shape[0]} students")
print()

# ============================================================
# STEP 3: EDA VISUALISATIONS → student_eda.png
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Student Academic Performance — Exploratory Analysis\n'
             'UCI Dataset (Realinho et al., 2021)', fontsize=14, fontweight='bold')

# [0,0] At-Risk distribution
counts = y.value_counts().sort_index()
axes[0, 0].bar(['Not At Risk', 'At Risk'], counts,
               color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
axes[0, 0].set_title('Class Distribution')
axes[0, 0].set_ylabel('Number of Students')
for i, v in enumerate(counts):
    axes[0, 0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# [0,1] 1st sem approved units by class
col_approved = 'Curricular units 1st sem (approved)'
for label, color, name in [(0, '#2ecc71', 'Not At Risk'), (1, '#e74c3c', 'At Risk')]:
    axes[0, 1].hist(df[df['at_risk'] == label][col_approved],
                    bins=15, alpha=0.6, color=color, label=name, edgecolor='white')
axes[0, 1].set_title('1st Sem Units Approved (by Risk)')
axes[0, 1].set_xlabel('Units Approved')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend()

# [1,0] Heatmap of top 10 features most correlated with at_risk
corr_with_target = df[FEATURE_COLS + ['at_risk']].corr()['at_risk'].drop('at_risk')
top10_cols = corr_with_target.abs().nlargest(10).index.tolist()
corr_matrix = df[top10_cols + ['at_risk']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=axes[1, 0], linewidths=0.5, square=False)
axes[1, 0].set_title('Top 10 Features — Correlation with At-Risk')

# [1,1] Scatter: 1st sem approved vs 1st sem grade
col_grade = 'Curricular units 1st sem (grade)'
colors_map = df['at_risk'].map({0: '#2ecc71', 1: '#e74c3c'})
axes[1, 1].scatter(df[col_approved], df[col_grade],
                   c=colors_map, alpha=0.4, edgecolors='none', s=18)
axes[1, 1].set_title('1st Sem: Units Approved vs Grade')
axes[1, 1].set_xlabel('Units Approved')
axes[1, 1].set_ylabel('Grade')
axes[1, 1].legend(handles=[Patch(color='#2ecc71', label='Not At Risk'),
                            Patch(color='#e74c3c', label='At Risk')])

plt.tight_layout()
plt.savefig('student_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: student_eda.png")

# ============================================================
# STEP 4: TRAIN 3 MODELS  (class_weight='balanced' on all)
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
    ),
}

# ============================================================
# STEP 5: EVALUATE ALL MODELS — primary metric: F1 for class 1
# ============================================================
results = {}
print(f"\n{'Model':<22} {'Acc':>6}  {'Prec(1)':>9}  {'Rec(1)':>8}  {'F1(1)':>7}")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'recall':    recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'f1':        f1_score(y_test, y_pred, pos_label=1, zero_division=0),
    }
    r = results[name]
    print(f"{name:<22} {r['accuracy']:>6.3f}  {r['precision']:>9.3f}  "
          f"{r['recall']:>8.3f}  {r['f1']:>7.3f}")

print("=" * 60)
print("KEY METRIC: Recall(1) — catching every at-risk student matters most.")
print("BEST MODEL: selected by F1(class 1), not raw accuracy.\n")

# Select best by F1 for the at-risk class
best_name  = max(results, key=lambda k: results[k]['f1'])
best       = results[best_name]
best_model = best['model']
y_pred     = best['y_pred']

print(f"Best Model  : {best_name}")
print(f"  Accuracy  : {best['accuracy']:.3f}")
print(f"  Precision : {best['precision']:.3f}  (of predicted at-risk, how many truly are)")
print(f"  Recall    : {best['recall']:.3f}  ← KEY: of all at-risk, how many we catch")
print(f"  F1 Score  : {best['f1']:.3f}")
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Not At Risk', 'At Risk'], zero_division=0))

# ============================================================
# STEP 6: CONFUSION MATRIX → confusion_matrix.png
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Not At Risk', 'At Risk']).plot(
    ax=ax, colorbar=False, cmap='Blues'
)
ax.set_title(f'Confusion Matrix — {best_name}', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

# ============================================================
# STEP 7: FEATURE IMPORTANCE → feature_importance.png (top 20)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

if best_name == 'Logistic Regression':
    importance = np.abs(best_model.coef_[0])
    imp_label  = 'Absolute Coefficient'
else:
    importance = best_model.feature_importances_
    imp_label  = 'Gini Importance'

top_n      = 20
sorted_idx = np.argsort(importance)
top_idx    = sorted_idx[-top_n:]

ax.barh([FEATURE_COLS[i] for i in top_idx], importance[top_idx],
        color='#3498db', edgecolor='black')
ax.set_title(f'Top {top_n} Feature Importances — {best_name} ({imp_label})',
             fontweight='bold')
ax.set_xlabel(imp_label)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

# ============================================================
# STEP 8: MODEL COMPARISON → model_comparison.png
# (Recall and F1, not raw accuracy — avoids misleading imbalance effect)
# ============================================================
model_names = list(results.keys())
recalls = [results[n]['recall'] for n in model_names]
f1s     = [results[n]['f1']     for n in model_names]
bar_colors = ['#e74c3c' if n == best_name else '#3498db' for n in model_names]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Model Comparison — At-Risk Class Metrics', fontsize=13, fontweight='bold')

for ax, values, title, ylabel in [
    (axes[0], recalls, 'Recall (At-Risk Class)', 'Recall'),
    (axes[1], f1s,     'F1 Score (At-Risk Class)', 'F1 Score'),
]:
    bars = ax.bar(model_names, values, color=bar_colors, edgecolor='black')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.15)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontweight='bold')
    ax.legend(handles=[Patch(color='#e74c3c', label='Best Model'),
                        Patch(color='#3498db', label='Other Models')])

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")

# ============================================================
# STEP 9: SAVE ARTIFACTS
# ============================================================
train_medians = X_train.median().to_dict()

joblib.dump(best_model,   'best_model.pkl')
joblib.dump(scaler,       'scaler.pkl')
joblib.dump(FEATURE_COLS, 'feature_names.pkl')
joblib.dump(train_medians,'feature_medians.pkl')

print("\nArtifacts saved:")
print("  best_model.pkl      — trained best model")
print("  scaler.pkl          — fitted StandardScaler")
print("  feature_names.pkl   — list of 35 feature column names")
print("  feature_medians.pkl — training-set medians for all 35 features")

# ============================================================
# STEP 10: EXAMPLE PREDICTION — struggling student profile
# ============================================================
print("\n" + "=" * 55)
print("  EXAMPLE PREDICTION — Struggling Student Profile")
print("=" * 55)

example = train_medians.copy()
example.update({
    'Curricular units 1st sem (approved)': 0,
    'Curricular units 1st sem (grade)':    0.0,
    'Curricular units 2nd sem (approved)': 0,
    'Curricular units 2nd sem (grade)':    0.0,
    'Tuition fees up to date':             0,
    'Debtor':                              1,
    'Age at enrollment':                   25,
    'Admission grade':                     100.0,
})

ex_df     = pd.DataFrame([example])[FEATURE_COLS]
ex_scaled = scaler.transform(ex_df)
pred      = best_model.predict(ex_scaled)[0]
prob      = best_model.predict_proba(ex_scaled)[0]

print(f"  Dropout Probability : {prob[1]*100:.1f}%")
print(f"  Prediction          : {'AT RISK' if pred == 1 else 'NOT AT RISK'}")
if pred == 1:
    print("  Recommended actions : counselling, payment plan, peer mentor")

print("\n" + "=" * 55)
print("  PROJECT COMPLETE — All outputs saved.")
print(f"  Best model: {best_name}")
print("=" * 55)
