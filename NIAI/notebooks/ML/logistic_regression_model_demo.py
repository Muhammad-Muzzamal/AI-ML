import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------------------------------------------
# 1. DATASET  (synthetic, so we KNOW the true rule behind the labels)
# ------------------------------------------------------------------
def make_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    study = rng.uniform(0, 10, n)          # hours studied
    sleep = rng.uniform(3, 9, n)           # hours slept
    # hidden "true" rule: more study & more sleep -> higher pass chance
    z = 1.2 * study + 0.5 * sleep - 8.0
    p = 1 / (1 + np.exp(-z))
    passed = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"study_hours": study, "sleep_hours": sleep, "passed": passed})

df = make_dataset()
df.to_csv("students.csv", index=False)
print(df.head(), "\n")
print("Class balance:\n", df["passed"].value_counts(), "\n")

# ------------------------------------------------------------------
# 2. DATA PIPELINE (sklearn)
# ------------------------------------------------------------------
X = df[["study_hours", "sleep_hours"]]
y = df["passed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),              # mean 0, std 1
    ("model", LogisticRegression()),
])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print("=== sklearn pipeline ===")
print("Test accuracy:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("Weights (scaled space):", pipe["model"].coef_[0])
print("Bias:", pipe["model"].intercept_[0], "\n")

# ------------------------------------------------------------------
# 3. FROM SCRATCH: how weights are found
# ------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def train_logreg(X, y, lr=0.1, epochs=1000, verbose=True):
    n, d = X.shape
    w = np.zeros(d)     # start with zero weights
    b = 0.0
    for epoch in range(epochs):
        z = X @ w + b               # 1) linear score
        p = sigmoid(z)              # 2) probability
        loss = log_loss(y, p)       # 3) how wrong are we?
        dw = X.T @ (p - y) / n      # 4) gradient wrt weights
        db = np.mean(p - y)         #    gradient wrt bias
        w -= lr * dw                # 5) step downhill
        b -= lr * db
        if verbose and epoch % 200 == 0:
            print(f"epoch {epoch:4d}  loss={loss:.4f}  w={np.round(w, 3)}  b={b:.3f}")
    return w, b

scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train)
Xte = scaler.transform(X_test)

print("=== from scratch (gradient descent) ===")
w, b = train_logreg(Xtr, y_train.values)
pred_scratch = (sigmoid(Xte @ w + b) >= 0.5).astype(int)
print("\nMy weights :", w, " bias:", b)
print("sklearn    :", pipe["model"].coef_[0], " bias:", pipe["model"].intercept_[0])
print("Test accuracy (scratch):", accuracy_score(y_test, pred_scratch))
# NOTE: sklearn adds L2 regularization (C=1.0) by default, so its weights
# come out slightly smaller than plain gradient descent. Same idea otherwise.

# Predict for a new student
new = pd.DataFrame({"study_hours": [6.0], "sleep_hours": [7.0]})
print("\nNew student pass probability:", pipe.predict_proba(new)[0, 1])