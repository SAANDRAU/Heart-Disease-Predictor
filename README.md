# Heart Disease Prediction 

## Overview

This project is a machine learning classification project that predicts the presence of heart disease based on patient health and clinical information.

The project includes data preprocessing, categorical encoding, handling of imbalanced data using SMOTE, feature scaling, comparison of multiple classification algorithms, hyperparameter tuning, model evaluation, model serialization using Pickle, and a Streamlit-based web interface for making predictions.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Streamlit
- Pickle

---

## Project Workflow

The project follows these main steps:

1. Data loading and exploration
2. Data preprocessing
3. Categorical feature encoding
4. Train-test splitting
5. Handling class imbalance using SMOTE
6. Feature scaling using MinMaxScaler
7. Training multiple classification models
8. Comparing model performance
9. Cross-validation
10. Hyperparameter tuning using GridSearchCV
11. Model evaluation
12. Saving the trained model and preprocessing components using Pickle
13. Building a Streamlit interface for prediction

---

## Data Preprocessing

The dataset contains a combination of numerical and categorical features.

Categorical features are converted into numerical values using `LabelEncoder`.

The data is divided into training and testing sets. SMOTE is used on the training data to address class imbalance.

`MinMaxScaler` is then used to scale the features before training the machine learning models.

---

## Machine Learning Models

The project compares several classification algorithms, including:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

The models are evaluated and compared to identify a suitable classification model.

---

## Hyperparameter Tuning

`GridSearchCV` is used to tune the Random Forest Classifier.

The hyperparameter search includes parameters such as:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

Cross-validation is used during the tuning process to identify a suitable combination of hyperparameters.

---

## Model Evaluation

The classification models are evaluated using:

- Accuracy
- Recall
- Classification Report
- Confusion Matrix
- Cross-validation

These metrics are used to understand how well the models perform on the classification task.

---

## Model Serialization

The selected trained model and preprocessing components are saved using Pickle.

The saved file contains:

- Trained machine learning model
- Label encoders
- MinMaxScaler

This allows the Streamlit application to reuse the trained model and the same preprocessing steps when making predictions on new input data.

---

## Streamlit Web Application

A Streamlit-based interface was developed for the project.

The application allows the user to enter patient-related information such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Angina
- Oldpeak
- ST Slope

The application encodes the categorical inputs, applies the saved scaler, and passes the processed data to the saved machine learning model.

The predicted result is then displayed in the Streamlit interface.

---

## Project Structure

```text
Heart-Disease-Prediction/
│
├── Heart_Disease_predictor.ipynb
├── Heart_disease_streamlit.py
├── Heart_Disease.pkl
├── README.md
└── dataset.csv
