"""
KNN (K-Nearest Neighbors) complete demo
1. Create a dataset (fruit classification: Apple / Orange / Mango)
2. KNN from scratch (numpy) + explain one prediction step by step
3. sklearn Pipeline (scale -> KNN)
4. Why scaling matters
5. Choosing k with cross-validation (+ plot)
6. GridSearchCV over k, weights, distance metric
"""
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# 1. DATASET
#    weight_g is deliberately on a much bigger scale than the others
#    and is NOT very useful for Apple vs Orange -> shows why scaling matters
# ------------------------------------------------------------------
def make_dataset(n_per_class=50, seed=7):
    rng = np.random.default_rng(seed)
    specs = {
        #          weight (mean, sd)  diameter_cm      sweetness (1-10)
        "Apple":  ((170, 25),         (7.0, 0.4),      (7.0, 0.7)),
        "Orange": ((175, 25),         (8.0, 0.4),      (5.0, 0.7)),
        "Mango":  ((300, 30),         (9.5, 0.6),      (8.5, 0.8)),
    }
    rows = []
    for label, (w, d, s) in specs.items():
        for _ in range(n_per_class):
            rows.append([rng.normal(*w), rng.normal(*d), rng.normal(*s), label])
    df = pd.DataFrame(rows, columns=["weight_g", "diameter_cm", "sweetness", "fruit"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True).round(2)


df = make_dataset()
df.to_csv("fruits.csv", index=False)
print(df.head(), "\n")
print(df.groupby("fruit").mean().round(2), "\n")

FEATURES = ["weight_g", "diameter_cm", "sweetness"]
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df["fruit"], test_size=0.25, random_state=0, stratify=df["fruit"]
)

# ------------------------------------------------------------------
# 2. KNN FROM SCRATCH
# ------------------------------------------------------------------
class KNNScratch:
    def __init__(self, k=5, metric="euclidean", weighted=False):
        self.k, self.metric, self.weighted = k, metric, weighted

    def fit(self, X, y):                      # KNN "training" = just memorise the data
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        return self

    def _distances(self, x):
        diff = self.X - x                     # difference to EVERY training point
        if self.metric == "euclidean":
            return np.sqrt((diff ** 2).sum(axis=1))
        if self.metric == "manhattan":
            return np.abs(diff).sum(axis=1)
        raise ValueError("metric must be 'euclidean' or 'manhattan'")

    def predict_one(self, x):
        d = self._distances(x)
        idx = np.argsort(d)[: self.k]         # indices of the k closest points
        labels = self.y[idx]
        if not self.weighted:                 # simple majority vote
            return Counter(labels).most_common(1)[0][0]
        votes = {}                            # closer neighbours get bigger vote
        for lbl, dist in zip(labels, d[idx]):
            votes[lbl] = votes.get(lbl, 0) + 1 / (dist + 1e-9)
        return max(votes, key=votes.get)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in np.asarray(X, dtype=float)])

    def explain(self, x):
        """Print the k neighbours and the vote for one query point."""
        d = self._distances(np.asarray(x, dtype=float))
        idx = np.argsort(d)[: self.k]
        table = pd.DataFrame(self.X[idx], columns=FEATURES)
        table["label"] = self.y[idx]
        table["distance"] = d[idx].round(3)
        print(table.to_string(index=False))
        print("Votes:", dict(Counter(self.y[idx])), "->", self.predict_one(x))


scaler = StandardScaler().fit(X_train)        # scale using TRAIN stats only
Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

print("=== KNN from scratch (k=5, scaled) ===")
knn = KNNScratch(k=5).fit(Xtr_s, y_train)
pred = knn.predict(Xte_s)
print("Test accuracy:", accuracy_score(y_test, pred), "\n")

print("Explaining one prediction (a 165 g, 7.1 cm, sweetness 7.2 fruit):")
query = scaler.transform(pd.DataFrame([[165, 7.1, 7.2]], columns=FEATURES))[0]
knn.explain(query)
print()

# ------------------------------------------------------------------
# 3. SKLEARN PIPELINE
# ------------------------------------------------------------------
pipe = Pipeline([("scaler", StandardScaler()),
                 ("knn", KNeighborsClassifier(n_neighbors=5))])
pipe.fit(X_train, y_train)
pred_sk = pipe.predict(X_test)
print("=== sklearn pipeline ===")
print("Test accuracy:", accuracy_score(y_test, pred_sk))
print("Confusion matrix (rows=true, cols=pred):\n",
      confusion_matrix(y_test, pred_sk, labels=["Apple", "Mango", "Orange"]))
print(classification_report(y_test, pred_sk))

# ------------------------------------------------------------------
# 4. WHY SCALING MATTERS
# ------------------------------------------------------------------
raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
print("=== Scaling effect (5-fold CV accuracy, k=5) ===")
cv_raw = cross_val_score(KNeighborsClassifier(5), df[FEATURES], df["fruit"], cv=5).mean()
cv_scaled = cross_val_score(pipe, df[FEATURES], df["fruit"], cv=5).mean()
print(f"Without scaling: {cv_raw:.3f}")
print(f"With scaling   : {cv_scaled:.3f}\n")

# ------------------------------------------------------------------
# 5. CHOOSING k WITH CROSS-VALIDATION
# ------------------------------------------------------------------
ks = list(range(1, 31))
scores = []
for k in ks:
    p = Pipeline([("scaler", StandardScaler()),
                  ("knn", KNeighborsClassifier(n_neighbors=k))])
    scores.append(cross_val_score(p, df[FEATURES], df["fruit"], cv=5).mean())

best_k = ks[int(np.argmax(scores))]
print("=== Choosing k ===")
print("k=1 :", round(scores[0], 3), "| k=5 :", round(scores[4], 3),
      "| k=15:", round(scores[14], 3), "| k=30:", round(scores[29], 3))
print("Best k:", best_k, "with CV accuracy", round(max(scores), 3), "\n")

plt.figure(figsize=(7, 4))
plt.plot(ks, scores, marker="o")
plt.axvline(best_k, color="red", linestyle="--", label=f"best k = {best_k}")
plt.xlabel("k (number of neighbours)")
plt.ylabel("5-fold CV accuracy")
plt.title("Choosing k")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("knn_k_vs_accuracy.png", dpi=120)

# ------------------------------------------------------------------
# 6. GRID SEARCH
# ------------------------------------------------------------------
grid = GridSearchCV(
    Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
    param_grid={
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],                     # 1 = Manhattan, 2 = Euclidean
    },
    cv=5,
)
grid.fit(X_train, y_train)
print("=== GridSearchCV ===")
print("Best params:", grid.best_params_)
print("Best CV score:", round(grid.best_score_, 3))
print("Held-out test accuracy:", round(grid.score(X_test, y_test), 3))
