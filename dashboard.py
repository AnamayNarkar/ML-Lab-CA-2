# ============================================================
#  STUDENT AT-RISK EARLY WARNING SYSTEM — Streamlit App
#  Run: streamlit run app.py
#
#  Dataset: Realinho et al. (2021), UCI ML Repository
#  DOI: https://doi.org/10.24432/C5MC89
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Student At-Risk Predictor",
    page_icon="🎓",
    layout="wide",
)

# ── Load model artifacts (cached so they don't reload on every widget change) ──
@st.cache_resource
def load_artifacts():
    model    = joblib.load('best_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    medians  = joblib.load('feature_medians.pkl')
    return model, scaler, features, medians

model, scaler, FEATURE_COLS, train_medians = load_artifacts()

MODEL_NAME  = type(model).__name__
DATASET_BASELINE = 32.1   # % dropout rate in UCI dataset

# ============================================================
# HEADER
# ============================================================
st.title("🎓 Student At-Risk Early Warning System")
st.markdown("""
**Problem Statement:** Predict academic performance to identify at-risk students early —
before they drop out — so educators can intervene in time.

- **At-Risk** = predicted to drop out
- **Dataset:** UCI *Predict Students' Dropout and Academic Success* — 4,424 students, 36 features (37 columns inc. target)
- **Citation:** Realinho et al. (2021). UCI ML Repository. https://doi.org/10.24432/C5MC89
""")
st.divider()

# ============================================================
# SIDEBAR — Student input form (top 10 most predictive features)
# ============================================================
st.sidebar.header("Student Profile")
st.sidebar.markdown("Enter the student's details. The remaining 26 features are filled with training-set medians automatically.")

inputs = {}

st.sidebar.subheader("Academic Performance")
inputs['Curricular units 1st sem (approved)'] = st.sidebar.slider(
    "1st Sem Units Approved", 0, 26,
    value=int(train_medians.get('Curricular units 1st sem (approved)', 5)),
    help="Number of curricular units the student passed in 1st semester"
)
inputs['Curricular units 1st sem (grade)'] = st.sidebar.slider(
    "1st Sem Average Grade", 0.0, 18.0,
    value=round(float(train_medians.get('Curricular units 1st sem (grade)', 12.0)), 1),
    step=0.1,
    help="Average grade in 1st semester (scale 0–18)"
)
inputs['Curricular units 2nd sem (approved)'] = st.sidebar.slider(
    "2nd Sem Units Approved", 0, 23,
    value=int(train_medians.get('Curricular units 2nd sem (approved)', 5)),
    help="Number of curricular units the student passed in 2nd semester"
)
inputs['Curricular units 2nd sem (grade)'] = st.sidebar.slider(
    "2nd Sem Average Grade", 0.0, 18.0,
    value=round(float(train_medians.get('Curricular units 2nd sem (grade)', 12.0)), 1),
    step=0.1,
    help="Average grade in 2nd semester (scale 0–18)"
)

st.sidebar.subheader("Enrollment & Finances")
inputs['Tuition fees up to date'] = st.sidebar.selectbox(
    "Tuition Fees Up to Date",
    options=[1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No",
    help="Whether the student's tuition is currently paid"
)
inputs['Scholarship holder'] = st.sidebar.selectbox(
    "Scholarship Holder",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
inputs['Debtor'] = st.sidebar.selectbox(
    "Has Outstanding Debt",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

st.sidebar.subheader("Background")
inputs['Age at enrollment'] = st.sidebar.slider(
    "Age at Enrollment", 17, 70,
    value=int(train_medians.get('Age at enrollment', 20))
)
inputs['Previous qualification (grade)'] = st.sidebar.slider(
    "Previous Qualification Grade", 95.0, 190.0,
    value=round(float(train_medians.get('Previous qualification (grade)', 133.0)), 1),
    step=0.1,
    help="Grade of previous qualification before university (95–190 scale)"
)
inputs['Admission grade'] = st.sidebar.slider(
    "Admission Grade", 95.0, 190.0,
    value=round(float(train_medians.get('Admission grade', 126.0)), 1),
    step=0.1,
    help="Grade at university admission (95–190 scale)"
)

predict_btn = st.sidebar.button(
    "Predict At-Risk Status", type="primary", use_container_width=True
)

# ============================================================
# MAIN PANEL — Results
# ============================================================
if predict_btn:
    # Build full 35-feature row: start from training medians, override top-10 inputs
    student_row = train_medians.copy()
    student_row.update(inputs)
    input_df = pd.DataFrame([student_row])[FEATURE_COLS]

    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    proba        = model.predict_proba(input_scaled)[0]
    risk_pct     = proba[1] * 100
    safe_pct     = proba[0] * 100

    # ── Result banner ─────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        if prediction == 1:
            st.error(f"⚠️  AT RISK — Dropout Probability: {risk_pct:.1f}%")
            st.markdown("""
**Recommended Interventions:**
- Schedule an academic counselling session
- Review and set up a tuition payment plan
- Assign a peer mentor for ongoing support
- Flag for attendance and grade monitoring
- Connect with student welfare services
""")
        else:
            st.success(f"✅  NOT AT RISK — Continuing Probability: {safe_pct:.1f}%")
            st.markdown("Student profile is within normal academic progression range.")

    with col2:
        st.metric(
            label="Dropout Risk",
            value=f"{risk_pct:.1f}%",
            delta=f"{risk_pct - DATASET_BASELINE:.1f}% vs avg",
            delta_color="inverse",
            help=f"Dataset average dropout rate: {DATASET_BASELINE}%"
        )
        st.metric(label="Safe Probability", value=f"{safe_pct:.1f}%")

    # ── Risk gauge ────────────────────────────────────────────
    if risk_pct > 60:
        gauge_color = "#e74c3c"
        risk_label  = "High Risk"
    elif risk_pct > 32:
        gauge_color = "#e67e22"
        risk_label  = "Moderate Risk"
    else:
        gauge_color = "#2ecc71"
        risk_label  = "Low Risk"

    st.markdown("**Risk Level:**")
    st.markdown(
        f'<div style="background: linear-gradient(to right, {gauge_color} {risk_pct:.0f}%, '
        f'#e0e0e0 {risk_pct:.0f}%); height: 28px; border-radius: 6px; '
        f'border: 1px solid #ccc;"></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{risk_label} — {risk_pct:.1f}% | Threshold: 50% | Dataset baseline: {DATASET_BASELINE}%")

    # ── Feature importance chart ──────────────────────────────
    st.subheader("What drove this prediction?")

    if hasattr(model, 'feature_importances_'):
        imp       = model.feature_importances_
        imp_label = 'Gini Importance'
    else:
        imp       = np.abs(model.coef_[0])
        imp_label = 'Absolute Coefficient'

    top_n      = 15
    sorted_idx = np.argsort(imp)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([FEATURE_COLS[i] for i in sorted_idx], imp[sorted_idx],
            color='#3498db', edgecolor='black')
    ax.set_xlabel(imp_label)
    ax.set_title(f'Top {top_n} Features — {MODEL_NAME} ({imp_label})', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

else:
    st.info("👈  Enter student details in the sidebar and click **Predict At-Risk Status**.")

    # Show EDA image if available
    import os
    if os.path.exists('student_eda.png'):
        st.subheader("Dataset Overview")
        st.image('student_eda.png', caption='UCI Dataset — Exploratory Analysis')

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Dataset: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). "
    "Predict Students' Dropout and Academic Success. UCI ML Repository. "
    "https://doi.org/10.24432/C5MC89  |  "
    f"Model: {MODEL_NAME}  |  Trained on 3,539 students, tested on 885 students."
)
