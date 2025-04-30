# Food Delivery Time Prediction

## Project Overview

This project aims to predict the **delivery time of food** using machine learning models. The project takes into account factors such as distance, preparation time, courier experience, weather conditions, and traffic levels to predict the time it will take for food to be delivered.

## Features

- **Data Preprocessing:** Handles missing values, encoding categorical variables, and feature scaling.
- **Exploratory Data Analysis (EDA):** Visualizes the dataset to gain insights into relationships between features.
- **Modeling:** Implements various regression models to predict delivery time, including:
    - Linear Regression
    - Random Forest Regressor
    - XGBoost
    - Support Vector Regression (SVR)
    - Decision Tree Regressor
    - K-Nearest Neighbors (KNN)
    - Lasso, Ridge, and ElasticNet
      
 
##  Usage


## 1.  Installation

To run this project, you need the following libraries:

- pandas
- numpy
- scikit-learn
- xgboost
- flask
- matplotlib
- seaborn
- pickle

## 2.  **Run the Complete Pipeline:** 
-The entire data analysis and model training process is included in a single file (food_delivery_time_prediction.py). This file performs the following tasks:

## **Libraries and Imports:** 
-All required libraries such as ** pandas** , **numpy**, **matplotlib**, **seaborn**, and **machine learning models** from **sklearn** and **xgboost** are imported at the beginning of the script.

## **Exploratory Data Analysis (EDA):**

* Statistical summaries of the dataset.

* Visualizations to analyze feature distributions and relationships (e.g., histograms, box plots).

## **Data Preprocessing:**

* Cleaning the data by handling missing values and encoding categorical features.

* Feature scaling for numerical columns and splitting the dataset into training and testing sets.

## **Handling Outliers:**

- Identifying and handling outliers using techniques like Z-score or IQR (Interquartile Range).

## **Model Training:**

- Training multiple machine learning models: Linear Regression, Random Forest, XGBoost, Decision Trees, SVR, KNN, Lasso, Ridge, and ElasticNet.

- Evaluation of models using metrics like MAE, MSE, RMSE, and R² score.

## **Model Selection:**

- The model with the best performance (lowest MAE) is saved for future use.

## **Testing on New Data:**

- The trained model is used to predict delivery times on new data. An example of this process is included in the script.
- **The script will:**
* Perform EDA, data cleaning, and preprocessing.
* Train and evaluate models.
* Save the best-performing model to a file (best_model.pkl).
* Predict delivery time for new data points based on user input.

## 3. **Model Prediction Example:** 
- After the model is trained, you can test it with new data points. Here's an example of how you can use the saved model for predictions:

import pickle
import pandas as pd

# Load the saved model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define new data (input your data in the correct format)
new_data = {
    'Distance_km': [5],
    'Preparation_Time_min': [12],
    'Courier_Experience_yrs': [4.0],
    'Weather_Foggy': [0],
    'Weather_Rainy': [1],
    # Add other features as required
}
new_df = pd.DataFrame(new_data)

# Predict delivery time
prediction = model.predict(new_df)
print("Estimated Delivery Time:", prediction[0])

## **Example README snippet for Usage section:**
 1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

