# ML Lab — All 10 Practicals
## Python Code + Theory + Viva Questions & Answers

---

## Quick Reference

| P# | Practical | Algorithm | Dataset |
|---|---|---|---|
| 1 | Uber fare prediction | Linear Regression | uber.csv |
| 2 | Uber fare prediction | Random Forest Regression | uber.csv |
| 3 | Email spam detection | K-Nearest Neighbors (KNN) | spam.csv |
| 4 | Email spam detection | Support Vector Machine (SVM) | spam.csv |
| 5 | Bank customer churn prediction | ANN (Neural Network) | churn.csv |
| 6 | Local minima of y=(x+3)² | Gradient Descent | — |
| 7 | Diabetes classification | KNN | diabetes.csv |
| 8 | Sales clustering | K-Means + Elbow Method | sample.csv |
| 9 | Titanic survival prediction | Random Forest / Logistic Regression | titanic.csv |
| 10 | Local minima of y=(x+5)² | Gradient Descent | — |

---

## Install All Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
```

---

## Evaluation Metrics Cheat Sheet

| Metric | Formula | What it means |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correct predictions |
| Precision | TP/(TP+FP) | Of predicted positives, how many are truly positive |
| Recall | TP/(TP+FN) | Of actual positives, how many were found |
| F1 Score | 2×P×R/(P+R) | Harmonic mean of Precision and Recall |
| R² Score | 1 - SS_res/SS_tot | Variance explained by model (1.0 = perfect) |
| RMSE | √(mean((y_pred - y_actual)²)) | Average prediction error in original units |
| MSE | mean((y_pred - y_actual)²) | Mean squared prediction error |

---

---

# Practical 1 — Uber Fare Prediction using Linear Regression

**Algorithm:** Linear Regression
**AIM:** Predict Uber ride fare using Linear Regression. Preprocess data, identify outliers, find correlations, train and evaluate model.

---

## Theory

**Linear Regression** finds the best-fit straight line through data points:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```
- `y` = predicted fare
- `β₀` = intercept
- `β₁...βₙ` = coefficients (learned from data)
- Minimizes **Sum of Squared Errors (SSE)**

---

## Code

```python
# P1_linear_regression_uber.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# =============================================
# STEP 1: LOAD & PRE-PROCESS DATA
# =============================================
# Option A: Load real dataset
# df = pd.read_csv("uber.csv")

# Option B: Generate sample data (use if no CSV)
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "distance_km":    np.random.uniform(1, 50, n),
    "pickup_hour":    np.random.randint(0, 24, n),
    "passenger_count":np.random.randint(1, 6, n),
    "is_weekend":     np.random.randint(0, 2, n),
})
# Fare formula: base + per km + surge at peak hours + noise
df["fare_amount"] = (2.5 + 1.8 * df["distance_km"]
                     + 0.5 * df["passenger_count"]
                     + 3 * df["is_weekend"]
                     + np.random.normal(0, 3, n))
# Inject a few outliers
df.loc[0:4, "fare_amount"] = [500, 450, 600, 520, 480]

print("Dataset Shape:", df.shape)
print(df.head())
print("\nBasic Statistics:")
print(df.describe())

# =============================================
# STEP 2: IDENTIFY OUTLIERS (IQR Method)
# =============================================
print("\nSTEP 2: Identifying Outliers...")

Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["fare_amount"] < lower_bound) | (df["fare_amount"] > upper_bound)]
print(f"Outliers found: {len(outliers)} records")
print(f"IQR = {IQR:.2f} | Lower: {lower_bound:.2f} | Upper: {upper_bound:.2f}")

# Remove outliers
df_clean = df[(df["fare_amount"] >= lower_bound) & (df["fare_amount"] <= upper_bound)]
print(f"Records after removing outliers: {len(df_clean)}")

# Boxplot to visualize outliers
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.boxplot(df["fare_amount"])
plt.title("Before Outlier Removal")
plt.subplot(1, 2, 2)
plt.boxplot(df_clean["fare_amount"])
plt.title("After Outlier Removal")
plt.tight_layout()
plt.savefig("p1_outliers.png")
plt.show()

# =============================================
# STEP 3: CORRELATION ANALYSIS
# =============================================
print("\nSTEP 3: Correlation Analysis...")

corr = df_clean.corr()
print("Correlation with fare_amount:")
print(corr["fare_amount"].sort_values(ascending=False))

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("p1_correlation.png")
plt.show()

# =============================================
# STEP 4: TRAIN LINEAR REGRESSION MODEL
# =============================================
print("\nSTEP 4: Training Linear Regression Model...")

features = ["distance_km", "pickup_hour", "passenger_count", "is_weekend"]
X = df_clean[features]
y = df_clean["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:<20}: {coef:.4f}")
print(f"  Intercept            : {model.intercept_:.4f}")

# =============================================
# STEP 5: EVALUATE MODEL
# =============================================
print("\nSTEP 5: Model Evaluation...")

y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse  = mean_squared_error(y_test, y_pred)

print(f"  R² Score : {r2:.4f}  (1.0 = perfect)")
print(f"  RMSE     : {rmse:.4f} (lower is better)")
print(f"  MSE      : {mse:.4f}")

# Actual vs Predicted plot
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color="steelblue")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect Prediction")
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare (Linear Regression)")
plt.legend()
plt.tight_layout()
plt.savefig("p1_actual_vs_predicted.png")
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("p1_residuals.png")
plt.show()

print("\nPractical 1 Complete!")
```

---

## Viva Questions & Answers

**Q1. What is Linear Regression?**
> Linear Regression models the relationship between a dependent variable (fare) and one or more independent variables (distance, time) by fitting a straight line. It finds the coefficients that minimize the Sum of Squared Errors between predicted and actual values.

**Q2. What is the equation of Linear Regression?**
> `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ` where β₀ is the intercept and β₁...βₙ are coefficients. For Uber: fare = 2.5 + 1.8×distance + 0.5×passengers + ...

**Q3. What is the Ordinary Least Squares (OLS) method?**
> OLS is the standard method to fit a linear regression line. It finds coefficients that minimize the sum of squared differences between predicted and actual values: minimize Σ(yᵢ - ŷᵢ)².

**Q4. What is R² Score and what does it mean?**
> R² (coefficient of determination) measures how much variance in the target variable is explained by the model. R²=1.0 means perfect prediction, R²=0 means the model is no better than predicting the mean. R²=0.85 means the model explains 85% of variance.

**Q5. What is RMSE and why is it preferred over MSE?**
> RMSE (Root Mean Square Error) = √(MSE). It is preferred because it is in the same units as the target variable. If fare is in rupees, RMSE gives error in rupees — interpretable. MSE gives rupees² which is hard to interpret.

**Q6. What is an outlier and how did you detect it?**
> An outlier is a data point significantly different from others. Detected using the IQR method: calculate Q1, Q3, IQR = Q3-Q1. Any value below Q1-1.5×IQR or above Q3+1.5×IQR is an outlier. Visualized using a boxplot.

**Q7. What is correlation and why is it important?**
> Correlation measures the linear relationship between two variables, ranging from -1 (perfect negative) to +1 (perfect positive). 0 means no linear relationship. Important for feature selection — features highly correlated with the target improve the model; features correlated with each other (multicollinearity) can hurt it.

**Q8. What is multicollinearity?**
> When two or more independent variables are highly correlated with each other. This makes individual coefficients unstable and hard to interpret. Detected using a correlation heatmap or VIF (Variance Inflation Factor).

**Q9. What are the assumptions of Linear Regression?**
> (1) Linearity — relationship between X and y is linear. (2) Independence — observations are independent. (3) Homoscedasticity — constant variance of residuals. (4) Normality of residuals. (5) No multicollinearity.

**Q10. What is a residual plot and what does it tell you?**
> A residual plot shows the difference between actual and predicted values. If residuals are randomly scattered around 0, the model is a good fit. Patterns (e.g., curved pattern) indicate the relationship is non-linear. Fan shapes indicate heteroscedasticity.

**Q11. What is train-test split and why do we use it?**
> We split the dataset (typically 80% train, 20% test) to evaluate how well the model generalizes to unseen data. Training on all data and testing on the same data gives artificially high accuracy (overfitting). The test set simulates real-world deployment.

**Q12. What is overfitting and underfitting?**
> Overfitting: model memorizes training data but performs poorly on test data (high train accuracy, low test accuracy). Underfitting: model is too simple to capture the pattern (low accuracy on both). Goal is the right balance (good generalization).

**Q13. What preprocessing steps did you perform?**
> (1) Handled missing values (dropna or fillna). (2) Removed outliers using IQR method. (3) Feature engineering (extracting hour, day from timestamp). (4) Dropped irrelevant columns (key, pickup_datetime). (5) Ensured fare_amount > 0 and passenger_count > 0.

**Q14. Why not use Linear Regression for classification?**
> Linear Regression predicts continuous values. For classification (spam/not spam), we need probabilities between 0 and 1. Linear Regression can predict values outside [0,1] and is sensitive to outliers in classification tasks. Use Logistic Regression for binary classification instead.

**Q15. What is the difference between Simple and Multiple Linear Regression?**
> Simple Linear Regression uses ONE independent variable: `y = β₀ + β₁x`. Multiple Linear Regression uses TWO or more: `y = β₀ + β₁x₁ + β₂x₂ + ...`. The Uber fare problem is Multiple Linear Regression since distance, time, and passenger count are all used.

---

---

# Practical 2 — Uber Fare Prediction using Random Forest

**Algorithm:** Random Forest Regression
**AIM:** Predict Uber fare using Random Forest. Same preprocessing as P1.

---

## Theory

**Random Forest** is an ensemble of decision trees:
- Builds N decision trees on random subsets of data (Bootstrap Aggregation = Bagging)
- Each tree uses a random subset of features at each split
- Final prediction = **average** of all trees (for regression)
- More robust than a single decision tree — reduces variance

---

## Code

```python
# P2_random_forest_uber.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# --- Generate sample data (same as P1) ---
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "distance_km":     np.random.uniform(1, 50, n),
    "pickup_hour":     np.random.randint(0, 24, n),
    "passenger_count": np.random.randint(1, 6, n),
    "is_weekend":      np.random.randint(0, 2, n),
})
df["fare_amount"] = (2.5 + 1.8 * df["distance_km"]
                     + 0.5 * df["passenger_count"]
                     + 3 * df["is_weekend"]
                     + np.random.normal(0, 3, n))
df.loc[0:4, "fare_amount"] = [500, 450, 600, 520, 480]

# =============================================
# STEP 1: PREPROCESSING + OUTLIER REMOVAL
# =============================================
Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df["fare_amount"] >= Q1 - 1.5*IQR) &
              (df["fare_amount"] <= Q3 + 1.5*IQR)]
print(f"Clean dataset size: {len(df_clean)}")

# =============================================
# STEP 2: CORRELATION
# =============================================
print("\nCorrelation with fare_amount:")
print(df_clean.corr()["fare_amount"].sort_values(ascending=False))

# =============================================
# STEP 3: TRAIN/TEST SPLIT
# =============================================
features = ["distance_km", "pickup_hour", "passenger_count", "is_weekend"]
X = df_clean[features]
y = df_clean["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================
# STEP 4: TRAIN RANDOM FOREST
# =============================================
print("\nTraining Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,    # 100 trees
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# =============================================
# STEP 5: EVALUATE — RF vs Linear Regression
# =============================================
# Also train Linear Regression for comparison
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n--- Model Comparison ---")
print(f"{'Model':<25} {'R² Score':>10} {'RMSE':>10}")
print("-" * 47)
for name, y_pred in [("Linear Regression", y_pred_lr),
                     ("Random Forest",     y_pred_rf)]:
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name:<25} {r2:>10.4f} {rmse:>10.4f}")

# Feature importance
print("\nFeature Importances (Random Forest):")
importances = pd.Series(rf_model.feature_importances_, index=features)
for feat, imp in importances.sort_values(ascending=False).items():
    bar = "#" * int(imp * 50)
    print(f"  {feat:<22}: {imp:.4f}  {bar}")

# Plot feature importances
plt.figure(figsize=(7, 4))
importances.sort_values().plot(kind="barh", color="steelblue")
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("p2_feature_importance.png")
plt.show()

# Actual vs Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color="green", label="RF")
plt.scatter(y_test, y_pred_lr, alpha=0.3, color="red",   label="LR")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "k--", lw=2, label="Perfect")
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("RF vs LR: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("p2_comparison.png")
plt.show()

print("\nPractical 2 Complete!")
```

---

## Viva Questions & Answers

**Q1. What is Random Forest?**
> Random Forest is an ensemble learning method that builds multiple decision trees during training and outputs the average prediction (regression) or majority vote (classification). Each tree is trained on a random bootstrap sample of data with a random subset of features at each split.

**Q2. What is Bagging (Bootstrap Aggregation)?**
> Bagging creates multiple training datasets by sampling with replacement from the original data. Each model trains on a different bootstrap sample. Final prediction is the average (regression) or majority vote (classification). Reduces variance and prevents overfitting.

**Q3. How does Random Forest differ from a single Decision Tree?**
> A single decision tree is prone to overfitting — it memorizes training data. Random Forest reduces overfitting by averaging many trees trained on random subsets of data and features. This reduces variance while keeping bias low.

**Q4. What is feature importance in Random Forest?**
> Feature importance measures how much each feature contributes to reducing impurity (MSE for regression) across all trees. Higher importance = more useful for prediction. Used for feature selection. Calculated as average reduction in impurity across all trees.

**Q5. What does `n_estimators` control?**
> The number of decision trees in the forest. More trees = better accuracy and stability but slower training and prediction. Typically 100–500 trees are used. After a certain point, adding more trees gives diminishing returns.

**Q6. Why is Random Forest better than Linear Regression for Uber fare?**
> Linear Regression assumes a linear relationship between features and fare — but real fare pricing may be non-linear (e.g., surge pricing at certain hours). Random Forest captures non-linear patterns and interactions between features automatically.

**Q7. What is Out-of-Bag (OOB) error?**
> Since each tree is trained on ~63% of data (bootstrap sample), the remaining ~37% (out-of-bag samples) can be used as a built-in validation set. OOB error is an unbiased estimate of generalization error without needing a separate test set.

**Q8. What are hyperparameters of Random Forest?**
> `n_estimators` (number of trees), `max_depth` (max tree depth), `max_features` (features considered at each split), `min_samples_split` (min samples to split a node), `min_samples_leaf` (min samples in a leaf node).

**Q9. Can Random Forest handle missing values?**
> Not natively in scikit-learn — missing values must be imputed before training. However, some implementations like in R can handle missing values using proximity-based imputation.

**Q10. What is the bias-variance tradeoff in Random Forest?**
> Individual deep decision trees have LOW bias but HIGH variance (overfit). By averaging many trees (bagging), Random Forest keeps the low bias but dramatically reduces variance — giving better generalization.

**Q11. How do you evaluate a regression model?**
> R² Score: proportion of variance explained (higher is better, max 1.0). RMSE: average prediction error in original units (lower is better). MAE: mean absolute error. These metrics together give a complete picture.

**Q12. When would you choose Linear Regression over Random Forest?**
> Linear Regression is preferred when: the relationship is genuinely linear, interpretability is required (you need to explain coefficients), the dataset is small, or training speed is critical. Random Forest is preferred for complex non-linear relationships and larger datasets.

**Q13. What is the difference between regression and classification in Random Forest?**
> In regression, the final prediction is the average of all tree predictions (continuous value). In classification, each tree votes for a class and the majority class wins (discrete label).

**Q14. What is cross-validation and why use it?**
> Cross-validation splits data into k folds, trains on k-1 folds and tests on the remaining fold, repeats k times. Gives a more reliable estimate of model performance than a single train-test split. Reduces the luck factor of one particular split.

**Q15. What is the difference between Random Forest and Gradient Boosting?**
> Random Forest: trees built in parallel, each independent, averaging reduces variance. Gradient Boosting: trees built sequentially, each one corrects the errors of the previous. Boosting typically achieves higher accuracy but is slower and more prone to overfitting.

---

---

# Practical 3 — KNN for Email Spam Detection

**Algorithm:** K-Nearest Neighbors (KNN)
**AIM:** Classify emails as Spam/Not Spam. Evaluate with Confusion Matrix, Accuracy, Precision, Recall.

---

## Theory

**KNN** classifies a new point by finding the K nearest training points and taking the majority vote:
1. Calculate distance (Euclidean) from new point to all training points
2. Select K nearest neighbors
3. Assign the class that appears most among the K neighbors

---

## Code

```python
# P3_knn_spam.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score,
                             f1_score, classification_report)
import warnings
warnings.filterwarnings("ignore")

# --- Generate sample spam data (replace with real dataset) ---
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "word_freq_free":    np.random.uniform(0, 5, n),
    "word_freq_money":   np.random.uniform(0, 4, n),
    "word_freq_click":   np.random.uniform(0, 3, n),
    "char_freq_excl":    np.random.uniform(0, 2, n),
    "capital_run_avg":   np.random.uniform(1, 50, n),
    "num_links":         np.random.randint(0, 10, n),
})
# Spam = 1 if high suspicious word frequency
df["label"] = ((df["word_freq_free"] + df["word_freq_money"] +
                df["word_freq_click"] + df["char_freq_excl"]) > 7).astype(int)
df["label"] += np.random.randint(0, 2, n)
df["label"] = (df["label"] > 0).astype(int)

print("Dataset shape:", df.shape)
print("Class distribution:")
print(df["label"].value_counts())
print(f"  0 = Not Spam: {(df['label']==0).sum()}")
print(f"  1 = Spam:     {(df['label']==1).sum()}")

# =============================================
# PREPROCESSING
# =============================================
X = df.drop("label", axis=1)
y = df["label"]

# Scale features — CRITICAL for KNN (distance-based)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# =============================================
# FIND OPTIMAL K (Elbow method for KNN)
# =============================================
print("\nFinding optimal K...")
k_range = range(1, 21)
train_scores, test_scores = [], []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

best_k = k_range[np.argmax(test_scores)]
print(f"Best K = {best_k} with test accuracy = {max(test_scores):.4f}")

plt.figure(figsize=(8, 4))
plt.plot(k_range, train_scores, "b-o", label="Train Accuracy")
plt.plot(k_range, test_scores,  "r-o", label="Test Accuracy")
plt.axvline(x=best_k, color="green", linestyle="--", label=f"Best K={best_k}")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("KNN: Accuracy vs K")
plt.legend()
plt.tight_layout()
plt.savefig("p3_knn_k_selection.png")
plt.show()

# =============================================
# TRAIN KNN WITH BEST K
# =============================================
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# =============================================
# EVALUATE
# =============================================
print("\n--- KNN Evaluation ---")

acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
error     = 1 - acc

print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Error Rate: {error:.4f}  ({error*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Spam", "Spam"],
            yticklabels=["Not Spam", "Spam"])
plt.title(f"KNN Confusion Matrix (K={best_k})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("p3_confusion_matrix.png")
plt.show()

print("\nPractical 3 Complete!")
```

---

## Viva Questions & Answers

**Q1. How does KNN work?**
> KNN stores all training data. For a new point, it calculates distance to all training points, finds K nearest neighbors, and assigns the majority class among those K neighbors. No explicit training phase — it is a lazy learner.

**Q2. What is the effect of K value on KNN?**
> Small K (K=1): captures fine patterns but is sensitive to noise (overfitting). Large K: smoother decision boundaries but may miss local patterns (underfitting). Optimal K is found by cross-validation or plotting accuracy vs K.

**Q3. Why must features be scaled before using KNN?**
> KNN uses Euclidean distance. If one feature has values in thousands (salary) and another in units (age), the distance will be dominated by the large-valued feature. StandardScaler ensures all features contribute equally to distance calculation.

**Q4. What is Euclidean distance?**
> `d = √(Σ(xᵢ - yᵢ)²)` — the straight-line distance between two points in n-dimensional space. Other distances used in KNN: Manhattan distance (|x₁-y₁| + |x₂-y₂|) and Minkowski distance (generalization of both).

**Q5. What is a Confusion Matrix?**
> A 2×2 table for binary classification: TP (true positive), TN (true negative), FP (false positive — spam predicted as not spam), FN (false negative — not spam predicted as spam). All four metrics are derived from this matrix.

**Q6. What is the difference between Precision and Recall?**
> Precision = TP/(TP+FP) — of all emails predicted as spam, how many are actually spam. Recall = TP/(TP+FN) — of all actual spam emails, how many were correctly identified. In spam detection, high Recall is more important (don't miss spam) but very low Precision means too many good emails flagged.

**Q7. What is F1 Score and when is it used?**
> F1 = 2×Precision×Recall / (Precision+Recall) — harmonic mean. Used when classes are imbalanced and you want to balance Precision and Recall. Better than accuracy when one class is rare (e.g., 95% not spam, 5% spam).

**Q8. What is a False Positive vs False Negative?**
> False Positive (FP): predicted spam but actually not spam — a legitimate email incorrectly flagged (annoying but less dangerous). False Negative (FN): predicted not spam but actually spam — a spam email reaches inbox (more dangerous for phishing).

**Q9. Why is KNN called a lazy learner?**
> Because it doesn't build a model during training — it just stores the training data. All computation happens at prediction time. Contrast with eager learners (like decision trees) which build a model during training and use it quickly for prediction.

**Q10. What is the time complexity of KNN prediction?**
> O(n×d) per query where n is the number of training samples and d is the number of features. For each new point, KNN calculates distance to all n training points — very slow for large datasets. Solutions: KD-Tree or Ball Tree data structures reduce this to O(d×log n).

**Q11. What is the accuracy vs error rate?**
> Accuracy = (TP+TN)/(TP+TN+FP+FN) — fraction of correct predictions. Error Rate = 1 - Accuracy — fraction of incorrect predictions. For spam detection with 90% accuracy, error rate = 10%.

**Q12. What is feature engineering for email spam detection?**
> Creating useful features from raw email text: word frequencies (how often "free", "money" appear), character frequencies (exclamation marks, dollar signs), capital letter usage (ALL CAPS = suspicious), number of links, sender domain reputation.

**Q13. What is stratified train-test split?**
> Stratified split ensures the class distribution in train and test sets matches the original dataset. Important for imbalanced datasets — without it, test set might have very few spam examples by chance.

**Q14. What is the curse of dimensionality and how does it affect KNN?**
> As dimensions (features) increase, distances between points become increasingly similar — all points appear equally distant. KNN loses its ability to find meaningful neighbors. Solution: dimensionality reduction (PCA) or feature selection before applying KNN.

**Q15. How would you improve KNN for spam detection?**
> Tune K using cross-validation. Use better features (TF-IDF for text). Apply dimensionality reduction (PCA). Use weighted KNN (closer neighbors have more influence). Balance classes using SMOTE if dataset is imbalanced.

---

---

# Practical 4 — SVM for Email Spam Detection

**Algorithm:** Support Vector Machine (SVM)
**AIM:** Classify emails as Spam/Not Spam using SVM.

---

## Theory

**SVM** finds the **optimal hyperplane** that maximally separates two classes:
- Maximizes the **margin** (distance between hyperplane and nearest points of each class)
- **Support vectors** are the training points closest to the hyperplane
- Uses **kernel trick** to handle non-linearly separable data (RBF, polynomial, linear kernels)

---

## Code

```python
# P4_svm_spam.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score,
                             f1_score, classification_report)
import warnings
warnings.filterwarnings("ignore")

# --- Generate sample spam data ---
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "word_freq_free":  np.random.uniform(0, 5, n),
    "word_freq_money": np.random.uniform(0, 4, n),
    "word_freq_click": np.random.uniform(0, 3, n),
    "char_freq_excl":  np.random.uniform(0, 2, n),
    "capital_run_avg": np.random.uniform(1, 50, n),
    "num_links":       np.random.randint(0, 10, n),
})
df["label"] = ((df["word_freq_free"] + df["word_freq_money"] +
                df["word_freq_click"] + df["char_freq_excl"]) > 7).astype(int)
df["label"] += np.random.randint(0, 2, n)
df["label"] = (df["label"] > 0).astype(int)

# PREPROCESSING
X = df.drop("label", axis=1)
y = df["label"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# TRAIN SVM
print("Training SVM with RBF Kernel...")
svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# EVALUATE
print("\n--- SVM Evaluation ---")
acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"  Accuracy  : {acc:.4f}")
print(f"  Error Rate: {1-acc:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n  TN={cm[0,0]} FP={cm[0,1]}\n  FN={cm[1,0]} TP={cm[1,1]}")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Not Spam","Spam"], yticklabels=["Not Spam","Spam"])
plt.title("SVM Confusion Matrix")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("p4_svm_confusion.png")
plt.show()

# Compare different kernels
print("\n--- Kernel Comparison ---")
print(f"{'Kernel':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 54)
for kernel in ["linear", "rbf", "poly", "sigmoid"]:
    clf = SVC(kernel=kernel, random_state=42)
    clf.fit(X_train, y_train)
    yp = clf.predict(X_test)
    print(f"{kernel:<12} {accuracy_score(y_test,yp):>10.4f} "
          f"{precision_score(y_test,yp):>10.4f} "
          f"{recall_score(y_test,yp):>10.4f} "
          f"{f1_score(y_test,yp):>10.4f}")

print("\nPractical 4 Complete!")
```

---

## Viva Questions & Answers

**Q1. What is SVM?**
> Support Vector Machine finds the optimal hyperplane that maximally separates classes with the largest margin. The margin is the distance between the hyperplane and the nearest training points (support vectors). Maximizing margin improves generalization.

**Q2. What are support vectors?**
> Support vectors are the training data points that lie closest to the decision boundary (hyperplane). They are the critical points — removing any other point doesn't change the hyperplane. The model depends entirely on these points.

**Q3. What is the kernel trick in SVM?**
> The kernel trick maps data to a higher-dimensional space where it becomes linearly separable without explicitly computing the transformation. Common kernels: Linear (no transformation), RBF/Gaussian (infinite dimensions), Polynomial, Sigmoid.

**Q4. What is the RBF (Radial Basis Function) kernel?**
> RBF kernel: `K(x,y) = exp(-γ||x-y||²)`. It maps data to infinite dimensions and can separate any distribution. `γ` controls the influence of individual training examples — high γ = tight boundaries (overfit), low γ = smooth boundaries.

**Q5. What is the C parameter in SVM?**
> C is the regularization parameter. High C: tries to classify all training points correctly (low bias, high variance — may overfit). Low C: allows some misclassifications for a wider margin (high bias, low variance — may underfit). Must be tuned using cross-validation.

**Q6. What is the difference between hard-margin and soft-margin SVM?**
> Hard-margin SVM: requires all points to be correctly classified with margin — only works for perfectly linearly separable data. Soft-margin SVM: allows some misclassifications using slack variables, controlled by parameter C. Real data almost always needs soft-margin.

**Q7. Why is SVM effective for email spam detection?**
> Email spam features (word frequencies) create high-dimensional spaces. SVM works well in high dimensions. The kernel trick handles non-linear patterns. SVM is memory efficient (only stores support vectors). Works well with text features (TF-IDF).

**Q8. SVM vs KNN — which is better for spam detection?**
> SVM: better for high-dimensional data, faster prediction O(support vectors), handles non-linearity with kernels. KNN: no training phase, simple to understand, but slow prediction O(n) and sensitive to irrelevant features. For spam, SVM is generally preferred for large-scale text classification.

**Q9. What is the difference between SVM for classification and regression?**
> SVC (Support Vector Classifier) finds hyperplane to separate classes. SVR (Support Vector Regression) finds a hyperplane within an epsilon-tube around which most training points lie. The goal shifts from maximizing margin to minimizing prediction error within the tube.

**Q10. What is a hyperplane?**
> In n-dimensional space, a hyperplane is a flat subspace of (n-1) dimensions. In 2D, it's a line. In 3D, it's a plane. In higher dimensions, it's a hyperplane. SVM finds the hyperplane that best separates the two classes with maximum margin.

**Q11. What is overfitting in SVM and how to prevent it?**
> Overfitting occurs with very high C or very high γ — the model fits training data perfectly but generalizes poorly. Prevention: tune C and γ using grid search with cross-validation. Use regularization (lower C). Choose appropriate kernel.

**Q12. What is the time complexity of SVM training?**
> O(n²) to O(n³) for training where n is the number of training samples — quadratic programming problem. SVM is slow to train on very large datasets. For large-scale text classification, LinearSVC (O(n)) is preferred.

**Q13. Can SVM handle multi-class classification?**
> SVM is inherently binary. For multi-class: One-vs-Rest (OvR): train one SVM per class (class vs all others). One-vs-One (OvO): train one SVM for every pair of classes (n(n-1)/2 classifiers). OvO is default in scikit-learn.

**Q14. What is the confusion matrix for a spam classifier?**
> TP: spam correctly classified as spam. TN: not spam correctly classified as not spam. FP: not spam incorrectly classified as spam (legitimate email in spam folder). FN: spam incorrectly classified as not spam (spam reaches inbox). For spam detection, minimizing FN is critical.

**Q15. What preprocessing is needed before SVM?**
> Feature scaling (StandardScaler or MinMaxScaler) — SVM is sensitive to feature magnitude. Handle missing values. For text: TF-IDF vectorization converts text to numbers. Remove highly correlated features. Balance classes if imbalanced.

---

---

# Practical 5 — ANN for Bank Customer Churn Prediction

**Algorithm:** Artificial Neural Network (ANN)
**AIM:** Predict if a customer will leave the bank using a Neural Network.

---

## Theory

**ANN** is inspired by the human brain. Key concepts:
- **Neurons** → Processing units organized in layers
- **Layers** → Input, Hidden (1 or more), Output
- **Weights & Biases** → Learned during training
- **Activation Functions** → ReLU (hidden), Sigmoid (output for binary)
- **Backpropagation** → Updates weights using gradient descent
- **Epochs** → Complete passes through training data

---

## Code

```python
# P5_ann_churn.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             classification_report)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")

# --- Generate sample churn data ---
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "credit_score":   np.random.randint(300, 850, n),
    "geography":      np.random.choice(["France","Germany","Spain"], n),
    "gender":         np.random.choice(["Male","Female"], n),
    "age":            np.random.randint(18, 70, n),
    "tenure":         np.random.randint(0, 10, n),
    "balance":        np.random.uniform(0, 250000, n),
    "num_of_products":np.random.randint(1, 5, n),
    "has_cr_card":    np.random.randint(0, 2, n),
    "is_active":      np.random.randint(0, 2, n),
    "salary":         np.random.uniform(20000, 200000, n),
})
# Churn = 1 if old, low credit, high balance, not active
df["exited"] = ((df["age"] > 45) &
                (df["credit_score"] < 500) &
                (df["is_active"] == 0)).astype(int)
df["exited"] = (df["exited"].astype(int) | np.random.binomial(1, 0.15, n)).clip(0,1)

print("Dataset shape:", df.shape)
print("Churn distribution:")
print(df["exited"].value_counts())

# PREPROCESSING
le_geo = LabelEncoder()
le_gen = LabelEncoder()
df["geography"] = le_geo.fit_transform(df["geography"])
df["gender"]    = le_gen.fit_transform(df["gender"])

X = df.drop("exited", axis=1)
y = df["exited"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# BUILD ANN
print("\nBuilding ANN model...")
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1,  activation="sigmoid")    # binary output
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# TRAIN
print("\nTraining ANN...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# PLOT TRAINING HISTORY
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend()
plt.tight_layout()
plt.savefig("p5_ann_training.png")
plt.show()

# EVALUATE
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print("\n--- ANN Evaluation ---")
acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Stayed","Churned"]))

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n  TN={cm[0,0]} FP={cm[0,1]}\n  FN={cm[1,0]} TP={cm[1,1]}")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Stayed","Churned"], yticklabels=["Stayed","Churned"])
plt.title("ANN Confusion Matrix — Churn Prediction")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("p5_ann_confusion.png")
plt.show()

print("\nPractical 5 Complete!")
```

---

## Viva Questions & Answers

**Q1. What is an Artificial Neural Network (ANN)?**
> ANN is a computational model inspired by the human brain. It consists of layers of interconnected neurons (nodes). Each connection has a weight. Neurons apply an activation function to the weighted sum of their inputs. Learning = adjusting weights using backpropagation.

**Q2. What are the layers in an ANN?**
> Input Layer: receives raw features (credit score, age, balance, etc.). Hidden Layers: extract patterns through weighted connections and activation functions. Output Layer: produces the final prediction (1 neuron with sigmoid for binary classification).

**Q3. What is backpropagation?**
> Backpropagation calculates the gradient of the loss function with respect to each weight using the chain rule of calculus. Gradients flow backward from output to input. Weights are then updated using gradient descent to minimize loss.

**Q4. What is the sigmoid activation function?**
> `σ(x) = 1/(1+e⁻ˣ)` — maps any real value to (0,1). Used in the output layer for binary classification as it gives probability. Problem: vanishing gradient for very large or small values.

**Q5. What is ReLU activation and why is it preferred?**
> ReLU (Rectified Linear Unit): `f(x) = max(0, x)`. Returns 0 for negative inputs, x for positive. Advantages: computationally fast, does not saturate for positive values (no vanishing gradient), helps sparse activation. Most popular choice for hidden layers.

**Q6. What is the vanishing gradient problem?**
> During backpropagation in deep networks, gradients are multiplied together through many layers. With sigmoid/tanh activations, these multiplied gradients become extremely small (vanish) — weights in early layers are barely updated. Solution: ReLU activation, batch normalization, residual connections.

**Q7. What is Dropout and why is it used?**
> Dropout randomly sets a fraction of neurons to zero during each training step. This prevents neurons from co-adapting and forces the network to learn redundant representations. Acts as a form of regularization — reduces overfitting. Dropout(0.3) means 30% of neurons are dropped each step.

**Q8. What is Binary Cross-Entropy loss?**
> `L = -(y×log(ŷ) + (1-y)×log(1-ŷ))`. Used for binary classification. Penalizes the model heavily when it is confident and wrong. For y=1 (churned), if model predicts ŷ=0.01, loss is very high.

**Q9. What is the Adam optimizer?**
> Adam (Adaptive Moment Estimation) combines Momentum (uses past gradients) and RMSProp (adapts learning rate per parameter). Faster convergence than SGD. Most popular optimizer for deep learning. Key hyperparameter: learning_rate (default 0.001).

**Q10. What is an epoch vs batch size?**
> Epoch: one complete pass through all training data. Batch size: number of samples processed before updating weights. 50 epochs with batch_size=32 means weights are updated (n/32) times per epoch, for 50 epochs total.

**Q11. What is customer churn and why is it important to predict?**
> Churn is when a customer stops using a service. Predicting churn allows banks to intervene proactively (special offers, personalized service) before the customer leaves. Acquiring a new customer costs 5-7x more than retaining an existing one.

**Q12. What is the difference between ANN and Logistic Regression?**
> Logistic Regression: single layer, one set of weights, linear decision boundary. ANN: multiple layers, non-linear decision boundaries, can model complex patterns. ANN is more powerful but needs more data and is harder to interpret.

**Q13. What metrics are most important for churn prediction?**
> Recall is critical — we want to catch as many churning customers as possible (minimize FN). A customer we wrongly identify as churning (FP) just gets a retention offer (harmless). Missing a churning customer (FN) means losing that customer.

**Q14. What is overfitting in neural networks and how to detect it?**
> Overfitting: training accuracy increases but validation accuracy decreases (or gap between them widens). Detected by plotting training vs validation accuracy/loss over epochs. Prevention: Dropout, L2 regularization, early stopping, more training data.

**Q15. What is the purpose of validation_split in model.fit()?**
> Reserves a fraction of training data (here 20%) as validation set during training. Used to monitor the model's performance on unseen data each epoch — helps detect overfitting early. Different from the test set (used only for final evaluation).

---

---

# Practical 6 — Gradient Descent: Local Minima of y = (x+3)²

**Algorithm:** Gradient Descent
**AIM:** Find local minimum of `y = (x+3)²` starting from x=2.

---

## Theory

**Gradient Descent** iteratively moves in the direction of steepest descent (negative gradient):
```
x_new = x_old - learning_rate × dy/dx
```
For `y = (x+3)²`:
- Derivative: `dy/dx = 2(x+3)`
- True minimum at x = -3 (where dy/dx = 0)

---

## Code

```python
# P6_gradient_descent.py
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# DEFINE FUNCTION AND DERIVATIVE
# =============================================
def f(x):
    return (x + 3) ** 2

def df(x):
    return 2 * (x + 3)    # derivative of (x+3)^2

# =============================================
# GRADIENT DESCENT ALGORITHM
# =============================================
def gradient_descent(start_x, learning_rate, iterations):
    x = start_x
    history = [x]
    for i in range(iterations):
        grad = df(x)
        x = x - learning_rate * grad
        history.append(x)
        if i < 10 or i % 10 == 0:
            print(f"  Iter {i+1:>4}: x = {x:>10.6f}  y = {f(x):>10.6f}  grad = {grad:>10.6f}")
    return x, history

# =============================================
# RUN GRADIENT DESCENT
# =============================================
print("=" * 55)
print("Function: y = (x+3)^2")
print("Derivative: dy/dx = 2(x+3)")
print("True minimum: x = -3, y = 0")
print("=" * 55)

start_x       = 2
learning_rate = 0.1
iterations    = 50

print(f"\nStarting at x = {start_x}")
print(f"Learning rate = {learning_rate}")
print(f"Iterations    = {iterations}")
print(f"\nGradient Descent Progress:")

final_x, history = gradient_descent(start_x, learning_rate, iterations)
print(f"\nFinal Result:")
print(f"  x = {final_x:.6f} (true minimum = -3.0)")
print(f"  y = {f(final_x):.6f} (true minimum = 0.0)")

# =============================================
# VISUALIZE
# =============================================
x_range = np.linspace(-8, 4, 300)
y_range = f(x_range)

plt.figure(figsize=(10, 5))

# Plot 1: Function with descent path
plt.subplot(1, 2, 1)
plt.plot(x_range, y_range, "b-", linewidth=2, label="y = (x+3)²")
x_hist = np.array(history)
plt.plot(x_hist, f(x_hist), "ro-", markersize=4, alpha=0.7, label="Gradient Descent Steps")
plt.plot(start_x, f(start_x), "g^", markersize=12, label=f"Start (x={start_x})")
plt.plot(-3, 0, "k*", markersize=15, label="Minimum (x=-3)")
plt.xlabel("x"); plt.ylabel("y = (x+3)²")
plt.title("Gradient Descent on y = (x+3)²")
plt.legend(); plt.grid(True, alpha=0.3)

# Plot 2: x value converging over iterations
plt.subplot(1, 2, 2)
plt.plot(history, "r-o", markersize=4)
plt.axhline(y=-3, color="k", linestyle="--", label="True minimum (x=-3)")
plt.xlabel("Iteration"); plt.ylabel("x value")
plt.title("Convergence of x Over Iterations")
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("p6_gradient_descent.png")
plt.show()

# =============================================
# COMPARE LEARNING RATES
# =============================================
print("\n--- Effect of Learning Rate ---")
print(f"{'Learning Rate':>15} {'Final x':>10} {'Final y':>12} {'Converged':>10}")
print("-" * 50)
for lr in [0.01, 0.1, 0.5, 0.9, 1.0, 1.1]:
    x = 2.0
    for _ in range(100):
        x = x - lr * df(x)
    converged = abs(x - (-3)) < 0.001
    status = "Yes" if converged else "DIVERGED"
    print(f"{lr:>15.2f} {x:>10.4f} {f(x):>12.6f} {status:>10}")

print("\nPractical 6 Complete!")
```

---

## Viva Questions & Answers

**Q1. What is Gradient Descent?**
> Gradient Descent is an optimization algorithm that iteratively moves in the direction of the negative gradient of a function to find its minimum. Update rule: `x = x - α × df/dx` where α is the learning rate.

**Q2. Why do we subtract the gradient?**
> The gradient points in the direction of steepest increase. To minimize the function, we move in the opposite direction (steepest descent) — hence we subtract. `new_x = old_x - learning_rate × gradient`.

**Q3. What is the learning rate and its effect?**
> The learning rate (α) controls the step size. Too small: very slow convergence, many iterations needed. Too large: overshoots the minimum and may diverge. Optimal: converges quickly and accurately. Typically tried as 0.001, 0.01, 0.1.

**Q4. What is the derivative of y = (x+3)² ?**
> Using chain rule: `dy/dx = 2(x+3)`. At x=2: derivative = 2(2+3) = 10. The gradient is positive (function is increasing), so gradient descent moves x to the left (x decreases).

**Q5. Where is the minimum of y = (x+3)²?**
> At x = -3, y = 0. Set `dy/dx = 0`: `2(x+3) = 0` → `x = -3`. Since the second derivative is positive (2 > 0), this is a minimum.

**Q6. What is local vs global minimum?**
> Local minimum: a point where the function value is lower than nearby points but may not be the lowest overall. Global minimum: the absolute lowest point. For y=(x+3)², there is only one minimum (x=-3) which is both local and global.

**Q7. What happens when learning rate = 1.0 for y=(x+3)²?**
> With lr=1.0 and derivative=2(x+3): `x_new = x - 1.0 × 2(x+3) = x - 2x - 6 = -x - 6`. Starting from x=2: x=-8, x=2, x=-8... it oscillates without converging.

**Q8. What is the difference between Gradient Descent, SGD, and Mini-batch GD?**
> Batch GD: uses all training data to compute gradient — accurate but slow. SGD (Stochastic): uses one random sample — fast but noisy. Mini-batch: uses a small batch (e.g., 32 samples) — balance of speed and accuracy. Most deep learning uses mini-batch.

**Q9. What is convergence in gradient descent?**
> Convergence means the algorithm has found a minimum (gradient ≈ 0). Detected when the change in x between iterations falls below a threshold (tolerance). For y=(x+3)², convergence means x ≈ -3.

**Q10. What are saddle points and how do they affect gradient descent?**
> A saddle point is where gradient = 0 but it's neither a minimum nor a maximum (like the center of a saddle). Gradient descent can get stuck at saddle points. Solutions: momentum, using adaptive optimizers like Adam.

**Q11. How many iterations did it take to converge?**
> Starting from x=2 with learning_rate=0.1, after 50 iterations x ≈ -3.0 (within numerical precision). The function converges exponentially — each step reduces the distance to minimum by factor (1-2α) = (1-0.2) = 0.8.

**Q12. What is the relationship between Gradient Descent and machine learning?**
> Machine learning training is fundamentally an optimization problem — minimize the loss function (e.g., MSE, cross-entropy) with respect to model weights. Gradient Descent (and variants like Adam, RMSProp) are how neural networks and linear models learn their parameters.

**Q13. What is momentum in gradient descent?**
> Momentum adds a fraction of the previous update to the current update: `velocity = β×velocity + gradient`, `x = x - α×velocity`. This accelerates convergence, helps escape local minima, and reduces oscillation. β=0.9 is typical.

**Q14. What is the difference between gradient descent and Newton's method?**
> Gradient descent uses only the first derivative (gradient). Newton's method uses both first and second derivatives (Hessian matrix). Newton's method converges faster but is computationally expensive for large parameter spaces.

**Q15. What would happen if the starting point x=2 was changed to x=-3?**
> At x=-3, the gradient = 2(-3+3) = 0. Gradient descent would immediately stop and report x=-3 as the minimum — which is correct! The gradient is zero at the minimum, so there's nothing to update.

---

---

# Practical 7 — KNN on Diabetes Dataset

**Algorithm:** K-Nearest Neighbors (KNN)
**AIM:** Predict diabetes using KNN. Compute confusion matrix, accuracy, error rate, precision, recall.

---

## Code

```python
# P7_knn_diabetes.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score,
                             f1_score, classification_report)
import warnings
warnings.filterwarnings("ignore")

# --- Load diabetes dataset ---
# Option A: from file
# df = pd.read_csv("diabetes.csv")

# Option B: from sklearn
from sklearn.datasets import load_diabetes
# Note: sklearn's diabetes is a regression dataset.
# Use the Pima Indians Diabetes dataset manually:
from sklearn.datasets import make_classification
X_raw, y_raw = make_classification(
    n_samples=768, n_features=8, n_informative=6,
    n_redundant=0, random_state=42
)
df = pd.DataFrame(X_raw, columns=[
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigree","Age"
])
df["Outcome"] = y_raw

print("Diabetes Dataset:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Outcome distribution:\n{df['Outcome'].value_counts()}")
print(f"  0 = No Diabetes: {(df['Outcome']==0).sum()}")
print(f"  1 = Diabetes:    {(df['Outcome']==1).sum()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic Stats:\n{df.describe().round(2)}")

# PREPROCESSING
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# FIND BEST K
k_scores = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    k_scores.append(knn.score(X_test, y_test))

best_k = np.argmax(k_scores) + 1
print(f"\nBest K = {best_k} with accuracy = {max(k_scores):.4f}")

# TRAIN WITH BEST K
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# EVALUATE
print("\n--- KNN Diabetes Evaluation ---")
acc       = accuracy_score(y_test, y_pred)
error     = 1 - acc
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Error Rate : {error:.4f}  ({error*100:.2f}%)")
print(f"  Precision  : {precision:.4f}")
print(f"  Recall     : {recall:.4f}")
print(f"  F1 Score   : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["No Diabetes","Diabetes"]))

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Diabetes","Diabetes"],
            yticklabels=["No Diabetes","Diabetes"])
plt.title(f"KNN Confusion Matrix (K={best_k}) — Diabetes")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("p7_diabetes_confusion.png")
plt.show()

print("\nPractical 7 Complete!")
```

---

## Key Viva Questions

**Q1. What is the Pima Indians Diabetes dataset?**
> A benchmark dataset with 768 records, 8 features (pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age) and a binary outcome (0=no diabetes, 1=diabetes). Originated from the National Institute of Diabetes and Digestive and Kidney Diseases.

**Q2. What is the significance of the Glucose feature for diabetes prediction?**
> Glucose is the strongest predictor of diabetes. High fasting glucose (> 126 mg/dL) is a primary diagnostic criterion for diabetes. The KNN model's feature importance (via permutation) would show Glucose as the top feature.

**Q3. What is error rate?**
> Error rate = 1 - Accuracy = (FP+FN)/(TP+TN+FP+FN). If accuracy is 75%, error rate is 25% — meaning 25% of predictions are wrong. Both metrics together give a complete picture.

**Q4. What is the medical importance of Recall in diabetes detection?**
> Recall (sensitivity) = TP/(TP+FN). In diabetes screening, we must minimize False Negatives (diabetics predicted as healthy). A missed diabetic patient doesn't receive treatment and faces serious complications. High recall is more critical than high precision.

**Q5. What is the diabetes pedigree function?**
> A feature in the Pima diabetes dataset that quantifies diabetes heredity. It considers family history (parents, siblings with diabetes) and their relation to the patient. Higher value = stronger genetic predisposition to diabetes.

*(Additional viva questions same as Practical 3 — KNN theory applies here too)*

---

---

# Practical 8 — K-Means Clustering on Sales Data

**Algorithm:** K-Means Clustering + Elbow Method
**AIM:** Cluster sales data and find optimal number of clusters using the Elbow Method.

---

## Theory

**K-Means** partitions data into K clusters:
1. Randomly initialize K centroids
2. Assign each point to the nearest centroid
3. Recalculate centroids as mean of assigned points
4. Repeat 2-3 until convergence
5. **Elbow Method**: plot WCSS vs K, choose K where curve bends (elbow)

**WCSS** (Within-Cluster Sum of Squares) = Σ(distance from each point to its centroid)²

---

## Code

```python
# P8_kmeans_sales.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# --- Generate sample sales data ---
np.random.seed(42)
n = 300

# 3 natural customer segments
seg1 = np.random.multivariate_normal([50000, 10], [[200000000, 5],[5, 4]], 100)  # high spenders
seg2 = np.random.multivariate_normal([20000, 25], [[50000000,  2],[2, 4]], 100)  # mid spenders
seg3 = np.random.multivariate_normal([5000,  50], [[5000000,   1],[1, 4]], 100)  # low spenders

data = np.vstack([seg1, seg2, seg3])
df = pd.DataFrame(data, columns=["Annual_Income", "Purchase_Frequency"])
df["Annual_Income"]       = df["Annual_Income"].clip(1000, 200000).round(-2)
df["Purchase_Frequency"]  = df["Purchase_Frequency"].clip(1, 60).round(0).astype(int)

print("Sales Dataset:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nBasic Stats:\n{df.describe().round(2)}")

# PREPROCESSING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =============================================
# ELBOW METHOD — Find optimal K
# =============================================
print("\nRunning Elbow Method...")
wcss = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(sil)
    print(f"  K={k}: WCSS={kmeans.inertia_:,.1f}  Silhouette={sil:.4f}")

# Plot Elbow curve
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(K_range, wcss, "bo-", markersize=8, linewidth=2)
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("WCSS (Inertia)")
axes[0].set_title("Elbow Method — Optimal K")
axes[0].axvline(x=3, color="red", linestyle="--", label="Optimal K=3")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, "go-", markersize=8, linewidth=2)
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs K")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("p8_elbow.png")
plt.show()

# =============================================
# TRAIN WITH OPTIMAL K=3
# =============================================
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", n_init=10, random_state=42)
kmeans.fit(X_scaled)
df["Cluster"] = kmeans.labels_

print(f"\nK-Means with K={optimal_k}:")
print(f"WCSS (Inertia): {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, kmeans.labels_):.4f}")

print("\nCluster Summary:")
cluster_summary = df.groupby("Cluster")[["Annual_Income","Purchase_Frequency"]].mean().round(0)
for cluster_id, row in cluster_summary.iterrows():
    print(f"  Cluster {cluster_id}: Income=₹{row['Annual_Income']:,.0f}, "
          f"Purchases/year={row['Purchase_Frequency']:.0f}")

# VISUALIZE CLUSTERS
plt.figure(figsize=(8, 6))
colors = ["red", "blue", "green"]
labels = ["Low Value", "Medium Value", "High Value"]
for i in range(optimal_k):
    mask = df["Cluster"] == i
    plt.scatter(df[mask]["Annual_Income"], df[mask]["Purchase_Frequency"],
                c=colors[i], label=f"Cluster {i}", alpha=0.6, s=50)

# Plot centroids (inverse transform)
centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids_orig[:,0], centroids_orig[:,1],
            c="black", marker="X", s=200, label="Centroids", zorder=5)

plt.xlabel("Annual Income (₹)")
plt.ylabel("Purchase Frequency (times/year)")
plt.title(f"K-Means Clustering (K={optimal_k}) — Customer Segments")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("p8_clusters.png")
plt.show()

print("\nPractical 8 Complete!")
```

---

## Viva Questions & Answers

**Q1. What is K-Means clustering?**
> K-Means partitions n data points into K clusters. Each point belongs to the cluster whose centroid (mean) is nearest. Algorithm: initialize K centroids, assign points to nearest centroid, recompute centroids, repeat until centroids stop moving.

**Q2. What is WCSS (Inertia)?**
> Within-Cluster Sum of Squares = sum of squared distances from each point to its cluster centroid. Lower WCSS = tighter, more compact clusters. Used in the Elbow Method to find optimal K.

**Q3. What is the Elbow Method?**
> Plot WCSS vs number of clusters K. As K increases, WCSS decreases (more clusters = tighter fit). The curve bends sharply at the optimal K — this "elbow" suggests the best K. Beyond this K, adding more clusters gives diminishing improvement.

**Q4. What is the Silhouette Score?**
> Measures how well each point fits its own cluster vs neighboring clusters. Range: -1 to +1. Near +1: point is well-matched to its cluster. Near 0: on the boundary between clusters. Near -1: likely in the wrong cluster. Optimal K has the highest silhouette score.

**Q5. What is K-Means++ initialization?**
> Standard K-Means uses random centroid initialization which can lead to poor convergence. K-Means++ chooses initial centroids that are far apart — first centroid randomly, each subsequent centroid chosen with probability proportional to its squared distance from the nearest existing centroid. Faster convergence and better results.

**Q6. What are the limitations of K-Means?**
> Must specify K in advance. Assumes spherical clusters (doesn't work well for elongated or irregularly shaped clusters). Sensitive to outliers (outliers can pull centroids). Sensitive to feature scaling (must standardize). May converge to local minimum.

**Q7. What is the difference between clustering and classification?**
> Classification is supervised — labels are known, model learns to classify new data. Clustering is unsupervised — no labels, algorithm discovers natural groupings. Classification predicts predefined categories; clustering discovers unknown patterns.

**Q8. What customer segments did the clustering reveal?**
> Typically: High-Value customers (high income, high purchase frequency) — premium segment. Medium-Value customers (moderate income, moderate purchases) — mainstream segment. Low-Value customers (low income, infrequent buyers) — budget segment. Used for targeted marketing.

**Q9. What is the difference between K-Means and K-Medoids?**
> K-Means: centroid is the mean of all points in the cluster (may not be an actual data point, sensitive to outliers). K-Medoids: centroid is the actual data point that minimizes total distance to others (more robust to outliers).

**Q10. What happens if K = N (number of data points)?**
> Every point becomes its own cluster. WCSS = 0 (no distance from point to its own centroid). This is perfect WCSS but useless — zero generalization. This is why the Elbow Method is needed to find a meaningful K.

**Q11. What is the time complexity of K-Means?**
> O(n × K × I × d) where n = samples, K = clusters, I = iterations, d = dimensions. Linear in n — scales well to large datasets. Typically converges in 10-100 iterations.

**Q12. When would you use Hierarchical Clustering instead of K-Means?**
> Hierarchical Clustering: when you don't know K in advance (dendogram reveals natural structure), when clusters are non-spherical, for small datasets, or when cluster hierarchy matters (taxonomy). K-Means: for large datasets, spherical clusters, when K is known.

**Q13. What preprocessing is necessary for K-Means?**
> Feature scaling is critical — K-Means uses Euclidean distance, so features with larger ranges dominate. Use StandardScaler (mean=0, std=1) or MinMaxScaler ([0,1]). Also handle missing values and outliers before clustering.

**Q14. What is the within-cluster vs between-cluster variance?**
> Good clustering = small within-cluster variance (points within a cluster are similar) and large between-cluster variance (different clusters are far apart). The Calinski-Harabasz index measures this ratio — higher is better.

**Q15. How do you validate clustering results without labels?**
> Internal validation: Silhouette Score, WCSS/Inertia, Davies-Bouldin Index (lower = better), Calinski-Harabasz Index (higher = better). External validation: if some labels exist, use Rand Index or Adjusted Rand Index to compare cluster assignments to true labels.

---

---

# Practical 9 — Titanic Survival Prediction

**Algorithm:** Random Forest / Logistic Regression Classifier
**AIM:** Predict passenger survival on Titanic using passenger features.

---

## Code

```python
# P9_titanic_survival.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score,
                             f1_score, classification_report)
import warnings
warnings.filterwarnings("ignore")

# --- Generate sample Titanic-like data ---
np.random.seed(42)
n = 891
df = pd.DataFrame({
    "Pclass":   np.random.choice([1,2,3], n, p=[0.24,0.21,0.55]),
    "Sex":      np.random.choice(["male","female"], n, p=[0.65,0.35]),
    "Age":      np.random.normal(30, 14, n).clip(1, 80),
    "SibSp":    np.random.choice([0,1,2,3], n, p=[0.68,0.23,0.07,0.02]),
    "Parch":    np.random.choice([0,1,2,3], n, p=[0.76,0.13,0.08,0.03]),
    "Fare":     np.random.lognormal(3, 1.2, n).clip(0, 500),
    "Embarked": np.random.choice(["S","C","Q"], n, p=[0.72,0.19,0.09])
})
# Survival: higher class, female, younger → more likely to survive
prob = (0.4 + 0.3*(df["Sex"]=="female").astype(int)
          - 0.1*(df["Pclass"]-1)
          - 0.001*df["Age"]).clip(0.05, 0.95)
df["Survived"] = np.random.binomial(1, prob)

print("Titanic Dataset:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Survival rate: {df['Survived'].mean():.2%}")

# EDA
print("\n--- Survival Analysis ---")
print("By Gender:")
print(df.groupby("Sex")["Survived"].mean().round(3))
print("\nBy Class:")
print(df.groupby("Pclass")["Survived"].mean().round(3))

# PREPROCESSING
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna("S", inplace=True)

le = LabelEncoder()
df["Sex"]      = le.fit_transform(df["Sex"])       # male=1, female=0
df["Embarked"] = le.fit_transform(df["Embarked"])

# Feature engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked",
            "FamilySize","IsAlone"]
X = df[features]
y = df["Survived"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# TRAIN MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

print("\n--- Model Comparison ---")
print(f"{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("-" * 50)

best_model = None
best_acc   = 0

for name, clf in models.items():
    clf.fit(X_train, y_train)
    yp = clf.predict(X_test)
    acc  = accuracy_score(y_test, yp)
    prec = precision_score(y_test, yp)
    rec  = recall_score(y_test, yp)
    f1   = f1_score(y_test, yp)
    print(f"{name:<22} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}")
    if acc > best_acc:
        best_acc   = acc
        best_model = clf
        best_name  = name
        best_preds = yp

print(f"\nBest model: {best_name}")
print("\nDetailed Report:")
print(classification_report(y_test, best_preds, target_names=["Did Not Survive","Survived"]))

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Did Not Survive","Survived"],
            yticklabels=["Did Not Survive","Survived"])
plt.title(f"{best_name} — Titanic Confusion Matrix")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("p9_titanic_confusion.png")
plt.show()

# Feature importance (Random Forest)
if hasattr(best_model, "feature_importances_"):
    plt.figure(figsize=(7, 4))
    feat_imp = pd.Series(best_model.feature_importances_, index=features)
    feat_imp.sort_values().plot(kind="barh", color="coral")
    plt.title("Feature Importances — Titanic")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("p9_titanic_importance.png")
    plt.show()

print("\nPractical 9 Complete!")
```

---

## Viva Questions & Answers

**Q1. What was the Titanic disaster?**
> The RMS Titanic sank on April 15, 1912, after hitting an iceberg. 1,502 of 2,224 passengers and crew died. The dataset contains passenger information (class, age, sex, fare) and whether they survived — used as a classic ML binary classification benchmark.

**Q2. What features most strongly predict survival?**
> Sex (women were evacuated first — "women and children first" policy), Pclass (1st class passengers had better access to lifeboats), Age (children had priority), Fare (correlated with class), FamilySize (moderate-sized families survived better than singles or very large families).

**Q3. What is feature engineering and give an example from Titanic?**
> Creating new features from existing ones. Examples: FamilySize = SibSp + Parch + 1 (total family members), IsAlone = 1 if FamilySize == 1 (travelling alone), Title extracted from Name (Mr, Mrs, Miss, Master), Fare_per_person = Fare / FamilySize.

**Q4. How did you handle missing values in the Titanic dataset?**
> Age: filled missing values with median age (robust to outliers). Embarked: 2 missing values filled with 'S' (most common port). Cabin: 77% missing — typically dropped or used as "has_cabin" binary feature. These choices affect model performance.

**Q5. What is Logistic Regression?**
> Logistic Regression is a classification algorithm that models the probability of a binary outcome using the sigmoid function: `P(y=1) = 1/(1+e^(-z))` where z = β₀ + β₁x₁ + ... Predicts 1 if P > 0.5, else 0. Despite its name, it is a classification algorithm not a regression algorithm.

**Q6. What is the difference between Logistic Regression and Linear Regression?**
> Linear Regression predicts continuous values. Logistic Regression predicts probabilities (0 to 1) for binary outcomes using the sigmoid function. Linear Regression minimizes SSE; Logistic Regression maximizes log-likelihood (or minimizes log-loss).

**Q7. How do you evaluate a binary classifier?**
> Confusion matrix (TP, TN, FP, FN). Accuracy, Precision, Recall, F1 Score. ROC-AUC curve (area under the ROC curve — higher is better). For imbalanced classes, accuracy alone is misleading.

**Q8. Why is accuracy alone not sufficient for Titanic?**
> The dataset is imbalanced (~62% did not survive, ~38% survived). A model that always predicts "did not survive" gets 62% accuracy without learning anything. Precision, Recall, and F1 Score give a more complete picture.

**Q9. What is stratified splitting and why is it used for Titanic?**
> Stratified splitting ensures the proportion of survived/not survived is the same in train and test sets as in the original data. Without it, you might get a test set with very few survivors by chance, giving misleading evaluation.

**Q10. What is the ROC curve and AUC?**
> ROC (Receiver Operating Characteristic) curve plots True Positive Rate (Recall) vs False Positive Rate at various threshold values. AUC (Area Under Curve) = 1.0 is perfect. AUC = 0.5 is random. Useful for imbalanced datasets — threshold-independent evaluation.

**Q11. How does the "women and children first" policy appear in the data?**
> Female survival rate (~74%) is much higher than male (~19%). Children (Age < 12) also have higher survival. This real-world knowledge guides feature engineering — Sex becomes the strongest predictor in the model.

**Q12. What is LabelEncoder and when should you use OHE instead?**
> LabelEncoder converts categorical text to integers (male=1, female=0). Suitable for ordinal categories or tree-based models. One-Hot Encoding (OHE) creates a binary column per category — better for nominal categories (like Embarked: S, C, Q) with linear models, as LabelEncoder implies a false ordering.

**Q13. What is cross-validation and how does it improve model evaluation?**
> k-fold cross-validation splits data into k folds, trains on k-1 and tests on 1, rotates through all folds. Gives a more reliable estimate of performance than a single 80/20 split. Reduces dependency on which samples happen to be in test set.

**Q14. What is the confusion matrix interpretation for Titanic survival?**
> TN: correctly predicted didn't survive. TP: correctly predicted survived. FP: predicted survived but actually didn't (false hope). FN: predicted didn't survive but actually did (more costly — missed a survivor). In humanitarian terms, minimizing FN (finding all actual survivors) is most important.

**Q15. How would you improve the Titanic model?**
> Extract title from Name (Mr, Miss, Dr, Master — reveals social status). Use Cabin deck as a feature (A-G decks had different positions on the ship). Create interaction features (Sex×Pclass). Try ensemble methods (XGBoost, LightGBM). Handle class imbalance (SMOTE). Use grid search for hyperparameter tuning.

---

---

# Practical 10 — Gradient Descent: Local Minima of y = (x+5)²

**Algorithm:** Gradient Descent
**AIM:** Find local minimum of `y = (x+5)²` starting from x=2.

---

## Code

```python
# P10_gradient_descent_v2.py
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x + 5) ** 2

def df(x):
    return 2 * (x + 5)    # derivative

def gradient_descent(start_x, lr, iters):
    x = start_x
    history = [x]
    print(f"{'Iter':>5} {'x':>12} {'y=f(x)':>12} {'gradient':>12}")
    print("-" * 45)
    for i in range(iters):
        grad = df(x)
        x = x - lr * grad
        history.append(x)
        if i < 15 or i % 10 == 0:
            print(f"{i+1:>5} {x:>12.6f} {f(x):>12.6f} {grad:>12.6f}")
    return x, history

print("=" * 50)
print("Function: y = (x+5)^2")
print("Derivative: dy/dx = 2(x+5)")
print("True minimum: x = -5, y = 0")
print("=" * 50)

start_x = 2
lr      = 0.1
iters   = 50

print(f"\nStarting point: x = {start_x}")
print(f"Learning rate : {lr}")

final_x, history = gradient_descent(start_x, lr, iters)
print(f"\nResult: x = {final_x:.6f}  (true = -5.0)")
print(f"        y = {f(final_x):.8f}  (true = 0.0)")

# VISUALIZE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x_range = np.linspace(-10, 5, 300)
axes[0].plot(x_range, f(x_range), "b-", lw=2, label="y = (x+5)²")
x_hist = np.array(history)
axes[0].plot(x_hist, f(x_hist), "ro-", markersize=5, alpha=0.7, label="GD Steps")
axes[0].plot(start_x, f(start_x), "g^", markersize=12, label=f"Start x={start_x}")
axes[0].plot(-5, 0, "k*", markersize=15, label="Minimum x=-5")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].set_title("Gradient Descent on y = (x+5)²")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history, "r-o", markersize=4)
axes[1].axhline(y=-5, color="k", linestyle="--", label="True min (x=-5)")
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("x value")
axes[1].set_title("Convergence of x")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("p10_gradient_descent.png")
plt.show()

# Learning rate comparison
print("\n--- Learning Rate Comparison ---")
print(f"{'LR':>8} {'Final x':>12} {'Iters to converge':>20}")
print("-" * 42)
for lr_try in [0.001, 0.01, 0.1, 0.5, 1.0, 1.2]:
    x = 2.0
    iters_conv = 0
    for i in range(1000):
        x = x - lr_try * df(x)
        iters_conv += 1
        if abs(x - (-5)) < 0.0001: break
    status = f"{iters_conv} iters" if abs(x-(-5)) < 0.01 else "DIVERGED"
    print(f"{lr_try:>8.3f} {x:>12.4f} {status:>20}")

print("\nPractical 10 Complete!")
```

---

## Key Differences from Practical 6

| Feature | P6: y = (x+3)² | P10: y = (x+5)² |
|---|---|---|
| Function | `(x+3)²` | `(x+5)²` |
| Derivative | `2(x+3)` | `2(x+5)` |
| True Minimum | x = -3 | x = -5 |
| Start point | x = 2 | x = 2 |
| Distance to travel | 5 units | 7 units |

> Same algorithm, different function. The code structure is identical — only `f(x)` and `df(x)` change. This demonstrates that gradient descent is a general optimization method applicable to any differentiable function.

---

## Viva Questions & Answers

**Q1. What is the derivative of y = (x+5)²?**
> `dy/dx = 2(x+5)`. At starting point x=2: gradient = 2(2+5) = 14. First step: x = 2 - 0.1×14 = 0.6. Gradient is always positive for x > -5, always negative for x < -5 — gradient descent always moves toward x=-5.

**Q2. How many iterations does it take to converge?**
> With lr=0.1, the update multiplier is (1-2×0.1)=0.8 per step. Distance reduces by 80% each iteration. Starting 7 units from minimum: after n steps, distance ≈ 7×(0.8)ⁿ. To get within 0.001: 7×(0.8)ⁿ < 0.001 → n > 40 iterations.

**Q3. What is the update equation for this specific problem?**
> `x_new = x_old - 0.1 × 2(x_old + 5) = x_old - 0.2(x_old + 5) = 0.8×x_old - 1`. Starting from x=2: 2 → 0.6 → -0.52 → -1.416 → -2.133 → ... → -5.

**Q4. What is the gradient at the minimum x=-5?**
> `df(-5) = 2(-5+5) = 2×0 = 0`. At the minimum, the gradient is exactly zero — gradient descent has nothing to update and stops (or makes zero-size steps). This is the stopping criterion.

**Q5. Compare P6 and P10 — what is the same and what is different?**
> Same: algorithm structure, derivative rule (2(x+c)), convergence behavior, effect of learning rate. Different: the shift constant (3 vs 5), the minimum location (-3 vs -5), the initial distance from minimum (5 vs 7 units). Both demonstrate gradient descent converges to the true minimum for convex functions.

*(All other viva questions same as Practical 6 — Gradient Descent theory applies fully)*

---

---

## Common Viva Questions Across All ML Practicals

**Q: What is supervised vs unsupervised learning?**
> Supervised: training data has labels (Practicals 1-5, 7, 9). Model learns mapping from features to labels. Unsupervised: no labels (Practical 8 — K-Means). Model finds structure/patterns in data. Gradient Descent (P6, P10) is an optimization technique used in both.

**Q: What is the bias-variance tradeoff?**
> Bias: error from wrong assumptions (underfitting). Variance: error from sensitivity to training data fluctuations (overfitting). Low bias + high variance = overfitting. High bias + low variance = underfitting. Goal: balance both for best generalization.

**Q: What is cross-validation?**
> k-fold cross-validation splits data into k folds. Trains on k-1 folds, tests on 1. Rotates k times. Final score = average. Gives reliable performance estimate. Reduces sensitivity to one particular train-test split.

**Q: What is regularization?**
> Technique to prevent overfitting by penalizing model complexity. L1 (Lasso): adds |coefficients| penalty — drives some coefficients to exactly 0 (feature selection). L2 (Ridge): adds coefficients² penalty — shrinks all coefficients toward 0. Elastic Net: combination of both.

**Q: What is feature scaling and why is it needed?**
> Standardizes feature ranges. StandardScaler: mean=0, std=1. MinMaxScaler: range [0,1]. Essential for: KNN (distance-based), SVM (kernel-based), ANN (gradient descent converges faster), Logistic Regression. Not needed for: Decision Trees, Random Forest (split-based, scale-invariant).

**Q: What is a confusion matrix?**
> For binary classification: 2×2 matrix. Rows = actual class. Columns = predicted class. TP=correctly predicted positive. TN=correctly predicted negative. FP=predicted positive but actually negative (Type I error). FN=predicted negative but actually positive (Type II error).

**Q: What is the difference between precision and recall?**
> Precision = TP/(TP+FP): of all positive predictions, how many are correct. Recall = TP/(TP+FN): of all actual positives, how many were found. Precision-Recall tradeoff: increasing threshold increases precision but decreases recall. Choose based on which error is costlier.
