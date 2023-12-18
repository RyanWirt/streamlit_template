import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Generate random data
np.random.seed(0)
num_samples = 100
X = np.random.rand(num_samples, 6)  # 6 random features
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(num_samples)  # Example linear relationship

# Create a DataFrame from the generated data
data = pd.DataFrame(data=X, columns=["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5", "Feature 6"])
data["Target"] = y

# Split the data into features (X) and target variable (y)
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Target variable (last column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model (you can add more evaluation metrics as needed)
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

print("Train R-squared:", train_score)
print("Test R-squared:", test_score)

# Save the trained model to a file
joblib.dump(rf_model, "../models/random_forest_model.pkl")
