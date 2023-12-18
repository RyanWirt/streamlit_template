import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load the pre-trained regression models
linear_model = joblib.load("../models/regression_model.pkl")
rf_model = joblib.load("../models/random_forest_model.pkl")

# Page title
st.title("Regression Analysis App")

# Sidebar with user inputs
st.sidebar.header("Input Features")

# Input fields for the six features
feature1 = st.sidebar.number_input("Feature 1", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
feature2 = st.sidebar.number_input("Feature 2", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
feature3 = st.sidebar.number_input("Feature 3", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
feature4 = st.sidebar.number_input("Feature 4", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
feature5 = st.sidebar.number_input("Feature 5", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
feature6 = st.sidebar.number_input("Feature 6", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

# Create a DataFrame with user input data
user_input = pd.DataFrame({
    "Feature 1": [feature1],
    "Feature 2": [feature2],
    "Feature 3": [feature3],
    "Feature 4": [feature4],
    "Feature 5": [feature5],
    "Feature 6": [feature6]
})

# Make predictions using the loaded models
linear_prediction = linear_model.predict(user_input)
rf_prediction = rf_model.predict(user_input)

# Display the predictions
st.write("Linear Regression Prediction:", linear_prediction[0])
st.write("Random Forest Prediction:", rf_prediction[0])
