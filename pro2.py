# salary_industry.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# 1. Load the enhanced dataset
data = pd.read_csv('Updated Salary_Data (2).csv')
print("First 5 rows:\n", data.head(), "\n")

# 2. Preprocess and feature engineering
data = data.dropna(subset=['Salary'])
X = data[['years_experience', 'age', 'job_title', 'education_level', 'gender']]
y = data['Salary']

numeric_features = ['years_experience', 'age']
categorical_features = ['job_title', 'education_level', 'gender']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 3. Model pipeline
pipeline = Pipeline([
    ('preproc', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=100, random_state=0))
])

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 5. Model training and hyperparameter tuning (small grid)
param_grid = {
    'reg__n_estimators': [100, 200],
    'reg__max_depth': [None, 10, 20]
}
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)

# 6. Evaluate
y_pred = grid.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {np.sqrt(mse):.2f}, R²: {r2:.2f}")

# 7. Feature importance (approximate)
# Use one-hot columns from the transformer
ohe = grid.best_estimator_.named_steps['preproc'].named_transformers_['cat']
feature_names = numeric_features + list(ohe.get_feature_names_out(categorical_features))
importances = grid.best_estimator_.named_steps['reg'].feature_importances_

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nTop features:\n", feat_imp.head())

# 8. Predict sample
sample = pd.DataFrame({
    'years_experience': [5],
    'age': [30],
    'job_title': ['Data Scientist'],
    'education_level': ['Master'],
    'gender': ['male']
})
print("\nSample Prediction:", grid.predict(sample)[0])

# 9. Simple plot: actual vs predicted
plt.figure(figsize=(6,6))
# plt.scatter(y_test, y_pred, alpha=0.3)
# plt.xlabel('Actual Salary')
# plt.ylabel('Predicted Salary')
# plt.title('Actual vs Predicted Salary')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.show()

plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.xlabel("Actual Salary (USD)")
plt.ylabel("Predicted Salary (USD)")
plt.title("Actual vs Predicted Salary")
plt.legend()
plt.show()
