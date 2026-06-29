# Simple Linear Regression — Complete Guide

A complete reference for understanding, implementing, and explaining Simple Linear Regression — including math intuition, code workflow, assumptions, limitations, real-world uses, and interview questions.

---

## 1. What is Simple Linear Regression?

Simple Linear Regression is a **supervised learning algorithm** used to model the relationship between **one independent variable (X)** and **one dependent variable (y)**, assuming that relationship is approximately a **straight line**.

It answers the question: *"If X changes, how much does y change, on average?"*

**Example (our dataset):**
`Salary = f(YearsExperience)`

---

## 2. The Mathematical Intuition

### 2.1 The Line Equation

Simple Linear Regression fits a line of the form:

```
y = β0 + β1*x + ε
```

| Symbol | Meaning |
|--------|---------|
| `y` | Dependent variable (target) — what we predict |
| `x` | Independent variable (feature) — what we use to predict |
| `β0` | Intercept — value of y when x = 0 |
| `β1` | Slope/Coefficient — change in y for a 1-unit change in x |
| `ε` | Error term — the part y that the line cannot explain (noise) |

In sklearn terms:
- `β0` → `model.intercept_`
- `β1` → `model.coef_`

### 2.2 How Do We Find the "Best" Line?

There are infinite possible lines we could draw through data. We need a way to measure which line is "best." That's where the **Cost Function** comes in.

### 2.3 Cost Function — Mean Squared Error (MSE)

```
MSE = (1/n) * Σ(y_actual - y_predicted)²
```

We square the errors because:
1. It removes negative signs (errors shouldn't cancel out)
2. It penalizes larger errors more heavily (robust to small noise, sensitive to big mistakes)

The "best" line is the one that **minimizes this MSE**.

### 2.4 Ordinary Least Squares (OLS) — The Closed-Form Solution

For simple linear regression, we don't need iterative optimization — there's a direct formula:

```
β1 = Σ((x_i - x̄)(y_i - ȳ)) / Σ((x_i - x̄)²)

β0 = ȳ - β1*x̄
```

Where `x̄` and `ȳ` are the means of x and y.

**Intuition:** β1 is essentially asking — *"How much does y move together with x, relative to how much x moves on its own?"* This is closely tied to **covariance** and **variance**:

```
β1 = Cov(x, y) / Var(x)
```

### 2.5 Gradient Descent (Alternative Approach)

For larger datasets or multiple features, instead of the closed-form OLS solution, we use **Gradient Descent** to iteratively minimize MSE:

```
β1 = β1 - α * ∂(MSE)/∂β1
β0 = β0 - α * ∂(MSE)/∂β0
```

Where `α` (alpha) is the **learning rate** — controls step size.

sklearn's `LinearRegression` uses OLS (closed-form) under the hood, not gradient descent, because it's exact and fast for this scale.

---

## 3. Evaluation Metrics

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **MAE** (Mean Absolute Error) | `(1/n)Σ\|y - ŷ\|` | Average absolute error, same unit as y |
| **MSE** (Mean Squared Error) | `(1/n)Σ(y - ŷ)²` | Penalizes large errors more |
| **RMSE** (Root MSE) | `√MSE` | Same unit as y, easier to interpret than MSE |
| **R² Score** (Coefficient of Determination) | `1 - (SS_res/SS_tot)` | % of variance in y explained by x (0 to 1, higher = better) |

**R² Intuition:**
- R² = 1 → model explains 100% of variance (perfect fit)
- R² = 0 → model explains nothing (same as predicting the mean every time)
- R² < 0 → model is worse than just predicting the mean

---

## 4. Assumptions of Linear Regression

For your model's results to be statistically valid (not just "it runs"), these assumptions should roughly hold:

1. **Linearity** — Relationship between X and y is actually linear
2. **Independence** — Observations are independent of each other
3. **Homoscedasticity** — Constant variance of residuals (errors don't get bigger/smaller as X increases)
4. **Normality of residuals** — Errors are normally distributed
5. **No (or minimal) multicollinearity** — Not really an issue in *simple* regression since there's only one feature; matters more in *multiple* regression

**How to check these:**
- Linearity → Scatter plot of X vs y
- Homoscedasticity → Residual plot (should look like random scatter, no funnel shape)
- Normality → Histogram/Q-Q plot of residuals

---

## 5. Step-by-Step Code Workflow

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv('salary_data.csv')

# 2. EDA
print(df.describe())
plt.scatter(df['YearsExperience'], df['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.show()

# 3. Train-test split
X = df[['YearsExperience']]   # Note: 2D for sklearn
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept (β0):", model.intercept_)
print("Slope (β1):", model.coef_[0])

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# 7. Visualize regression line
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

---

## 6. Real-World Uses

| Domain | Use Case |
|--------|----------|
| Business | Predicting sales based on advertising spend |
| HR | Predicting salary based on experience |
| Real Estate | Predicting house price based on size (sqft) |
| Healthcare | Predicting blood pressure based on age |
| Finance | Predicting stock trend based on a single economic indicator |
| Agriculture | Predicting crop yield based on rainfall |

Simple linear regression is rarely the *final* production model in complex problems, but it's the **first baseline model** almost every data scientist builds — if a simple line already explains most of the variance, you may not need anything fancier.

---

## 7. Limitations

1. **Only works for linear relationships** — Many real relationships are curved/exponential/etc. (e.g., diminishing returns)
2. **Sensitive to outliers** — A single extreme point can drastically shift the line, since OLS squares errors
3. **Only one feature** — Real-world problems usually depend on multiple factors (→ this is why **Multiple Linear Regression** exists)
4. **Assumes constant variance & normal errors** — Violations reduce reliability of confidence intervals/p-values
5. **Extrapolation risk** — Predicting far outside the training data range (e.g., 50 years experience when data only had 0–15) gives unreliable results
6. **Correlation ≠ Causation** — A strong linear relationship doesn't mean X *causes* y

---

## 8. Common Interview Questions

**Q1: What is the difference between correlation and regression?**
> Correlation measures the *strength and direction* of a relationship (a single number, -1 to 1) and doesn't imply prediction. Regression goes further — it builds an equation that lets you *predict* y from x.

**Q2: Why do we square the errors instead of taking absolute value in the cost function?**
> Squaring makes the function differentiable everywhere (needed for gradient-based optimization) and penalizes large errors more heavily. Absolute error (MAE) is non-differentiable at 0, which complicates optimization, though it's more robust to outliers.

**Q3: What does the slope (β1) represent?**
> The average change in y for a one-unit increase in x.

**Q4: What does it mean if β1 = 0?**
> X has no linear relationship with y — knowing x doesn't help predict y.

**Q5: How is R² different from Adjusted R²?**
> R² always increases (or stays the same) when you add more features, even useless ones. Adjusted R² penalizes for adding features that don't actually help — more relevant for multiple regression, since simple regression only has one feature anyway.

**Q6: What happens if you don't do a train-test split?**
> You risk overestimating your model's performance (overfitting) because you're evaluating it on the same data it learned from — it has effectively "seen the answers."

**Q7: Why use RMSE instead of MSE?**
> RMSE is in the same units as y (e.g., dollars), making it directly interpretable, whereas MSE is in squared units, making it harder to interpret in real-world terms.

**Q8: Can Simple Linear Regression handle categorical variables?**
> Not directly — categorical variables need encoding (e.g., one-hot encoding) first, and at that point you typically move into multiple regression since you now have more than one input dimension.

**Q9: What's the difference between simple and multiple linear regression?**
> Simple regression has one independent variable; multiple regression has two or more. The math generalizes using matrix notation: `y = Xβ + ε` where X becomes a matrix instead of a vector.

**Q10: How do you detect if linear regression is a good fit for your data?**
> Plot X vs y first. If the scatter plot doesn't look roughly like a straight line, linear regression isn't appropriate — you may need polynomial regression, log transforms, or a different algorithm entirely.

**Q11: What's the difference between OLS and Gradient Descent for finding coefficients?**
> OLS gives an exact closed-form solution by solving equations directly. Gradient Descent iteratively approaches the solution step by step. For simple/small datasets, OLS is faster and exact. For very large datasets or multiple regression with many features, gradient descent (or variants like SGD) scales better.

**Q12: What is heteroscedasticity, and why is it a problem?**
> It's when the variance of residuals is NOT constant across all values of X (e.g., errors get bigger as X increases). It violates a core regression assumption and means your confidence intervals/p-values become unreliable, even if your coefficient estimates are technically still unbiased.

---

## 9. Quick Summary Cheat Sheet

| Concept | One-liner |
|---------|-----------|
| Goal | Fit a straight line that best predicts y from x |
| Cost Function | Mean Squared Error (MSE) |
| Solving Method | OLS (closed-form) or Gradient Descent |
| Key Outputs | Intercept (β0), Slope (β1) |
| Best Metric to Report | RMSE (interpretable) + R² (explains variance) |
| Biggest Weakness | Assumes linearity, sensitive to outliers |
| When to Use | Quick baseline, one strong predictor, interpretability matters |
| When NOT to Use | Multiple strong predictors, non-linear pattern, lots of outliers |
