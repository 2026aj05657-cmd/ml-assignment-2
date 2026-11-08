import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "target"
df = pd.read_csv("heart.csv").drop_duplicates()
X, y = df.drop(columns=[TARGET]), df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
pd.concat([X_te, y_te], axis=1).to_csv("test_data.csv", index=False)

scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=7),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=7),
}
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.joblib")
joblib.dump(list(X.columns), "model/features.joblib")

rows = []
for name, m in models.items():
    m.fit(X_tr_s, y_tr)
    pred, prob = m.predict(X_te_s), m.predict_proba(X_te_s)[:, 1]
    rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred),
                 "AUC": roc_auc_score(y_te, prob), "Precision": precision_score(y_te, pred),
                 "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred),
                 "MCC": matthews_corrcoef(y_te, pred)})
    joblib.dump(m, f"model/{name.replace(' ', '_').lower()}.joblib")

res = pd.DataFrame(rows).round(4)
print(res.to_string(index=False))
res.to_csv("model/metrics.csv", index=False)
