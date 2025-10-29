# 📊 CTR Prediction Modeling — MIT xPRO

### Project Overview
This project focuses on developing predictive models to estimate **Click-Through Rate (CTR)** using anonymized digital advertising data.  
The objective was to build, compare, and evaluate multiple regression-based models to identify which algorithm best generalizes unseen data while maintaining interpretability and efficiency.

---

### 🎯 Objectives
- Predict CTR values for new ad impressions using historical training data.
- Compare model performance across **Linear Regression**, **Decision Tree Regressor**, and **XGBoost**.
- Evaluate results using **out-of-sample metrics (OSR², MAE, RMSE)**.
- Apply **feature encoding** and **complexity pruning** techniques to enhance model accuracy.

---

### 🧠 Methodology

1. **Data Preparation**
   - Imported pre-split datasets (`train.csv`, `test.csv`), representing 70% training and 30% testing data.
   - Extracted the target variable (`CTR`) and applied one-hot encoding to categorical features (`age`, `gender`).
   - Ensured data consistency between train and test sets.

2. **Model Development**
   - Implemented multiple algorithms:
     - **Linear Regression** — established baseline model.
     - **Decision Tree Regressor** — tested with full depth and pruned via *cost-complexity pruning path*.
     - **XGBoost** — used for advanced gradient-boosted performance.
   - Measured model performance using:
     - **MAE** (Mean Absolute Error)
     - **RMSE** (Root Mean Squared Error)
     - **OSR²** (Out-of-Sample R-squared)

3. **Model Evaluation**
   - Visualized model performance using custom evaluation functions.
   - Compared metrics before and after pruning to demonstrate the trade-off between bias and variance.
   - Selected the model with the highest **OSR²**, indicating the best predictive generalization.

---

### 🧰 Tech Stack

| Category | Tools & Libraries |
|-----------|------------------|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Modeling | Scikit-learn, XGBoost |
| Evaluation | MAE, RMSE, OSR² |
| Environment | Google Colab |

---

### 📈 Key Results
- **Linear Regression** established baseline interpretability.
- **Decision Tree Regressor** initially overfit but improved significantly after pruning.
- **XGBoost** achieved the highest **predictive accuracy** and best OSR² score across all models.
- Implementing complexity control and robust evaluation metrics resulted in a **notable improvement in out-of-sample performance**.

---

### 🧩 Example Workflow
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Train model
tree = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)

# Predict and evaluate
y_pred = tree.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
