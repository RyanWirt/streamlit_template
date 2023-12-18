import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Generate some sample data (replace with your dataset)
np.random.seed(0)
X = np.random.rand(100, 6)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)  # Example linear relationship

# Train a regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file
pickle.dump(model, open("../models/regression_model.pkl", "wb"))
