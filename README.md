Project Overview

This project focuses on predicting oral cancer diagnosis using machine learning techniques. It acts as a decision-support system to assist in early detection and analysis of risk factors associated with oral cancer.

Objectives
* Analyze patient data to predict oral cancer diagnosis.
* Identify major risk factors contributing to oral cancer.
* Compare performance of multiple machine learning models.
* Provide insights for early detection and preventive healthcare.

Key Features

* Oral cancer prediction using multiple ML models:
    * Logistic Regression
    * Decision Tree
    * Random Forest
* Data preprocessing including:
    * Handling missing values
    * One-hot encoding of categorical features
    * Feature scaling using StandardScaler
* Model evaluation using:
   * Accuracy, Precision, Recall, F1-Score
   * AUC-ROC score
   * Cross-validation
* Data visualization:
   * Target variable distribution
   * Histograms for numerical features
   * Count plots for categorical features
   * Confusion matrix visualization

Technology Stack

* Python
* Pandas (Data handling)
* NumPy (Numerical operations)
* Scikit-learn (Machine Learning models & evaluation)
* Matplotlib & Seaborn (Data visualization)

Dataset Description

* Dataset: Oral Cancer Prediction Dataset
* Contains ~84,000+ records with 25 features
* Includes:
    * Demographic data (Age, Gender, Country)
    * Lifestyle factors (Tobacco use, Alcohol consumption, Betel quid use)
    * Medical indicators (HPV infection, Oral lesions, Tumor size, Cancer stage)
    * Treatment and survival information
* Target Variable:
    * Oral Cancer (Diagnosis) (Yes/No)

Data Preprocessing

* Removed or handled missing values:
    * Numerical → filled with mean
    * Categorical → filled with mode
* Applied one-hot encoding for categorical variables
* Scaled numerical features using StandardScaler
* Converted target variable into binary format (0/1)

Machine Learning Models Used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

Model Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* AUC-ROC
* Cross-validation (5-fold)

Expected Outcomes

* Accurate prediction of oral cancer diagnosis
* Identification of important risk factors
* Comparative analysis of ML models
* Better understanding of healthcare data patterns

Results

* All models achieved very high accuracy (close to 100%)
* Indicates strong patterns in dataset
* Random Forest and Decision Tree showed excellent performance

Future Enhancements

* Add feature importance visualization
* Implement ROC curve comparison
* Build a web app using Flask/Streamlit
* Integrate real-time prediction system
* Use deep learning for improved generalization
* Apply model on real clinical datasets
