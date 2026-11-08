import joblib, pandas as pd, streamlit as st
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)

st.set_page_config(page_title="Heart Disease Classifier", page_icon="🫀", layout="wide")
st.title("🫀 Heart Disease Prediction – Model Evaluation")
st.caption("Upload test data, choose a model, see metrics. Models are pre-trained (no retraining).")

MODELS = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
TARGET = "target"

@st.cache_resource
def load(name):
    return joblib.load(f"model/{name.replace(' ', '_').lower()}.joblib")

scaler = joblib.load("model/scaler.joblib")
features = joblib.load("model/features.joblib")

with st.sidebar:
    file = st.file_uploader("Upload test CSV", type="csv")
    choice = st.selectbox("Select model", MODELS)

if file is None:
    st.info("Upload test_data.csv from the repo to begin.")
    st.stop()

df = pd.read_csv(file)
missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

X, y = scaler.transform(df[features]), df[TARGET]
model = load(choice)
pred, prob = model.predict(X), model.predict_proba(X)[:, 1]

st.subheader(f"Results: {choice}")
metrics = {"Accuracy": accuracy_score(y, pred), "AUC": roc_auc_score(y, prob),
           "Precision": precision_score(y, pred), "Recall": recall_score(y, pred),
           "F1": f1_score(y, pred), "MCC": matthews_corrcoef(y, pred)}
for c, (k, v) in zip(st.columns(6), metrics.items()):
    c.metric(k, f"{v:.3f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Reds", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification Report**")
    st.dataframe(pd.DataFrame(classification_report(y, pred, output_dict=True)).T.round(3))

with st.expander("Compare all models"):
    rows = []
    for n in MODELS:
        m = load(n); p = m.predict(X); pr = m.predict_proba(X)[:, 1]
        rows.append([n, accuracy_score(y, p), roc_auc_score(y, pr), precision_score(y, p),
                     recall_score(y, p), f1_score(y, p), matthews_corrcoef(y, p)])
    st.dataframe(pd.DataFrame(rows, columns=["Model","Accuracy","AUC","Precision","Recall","F1","MCC"]).round(4))
