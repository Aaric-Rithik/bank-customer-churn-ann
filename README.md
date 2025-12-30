# Bank Customer Churn Prediction (ANN)

This project predicts whether a bank customer is likely to churn using an Artificial Neural Network (ANN).

## Overview
Customer churn is an important business problem for banks. By analysing customer details such as age, balance, tenure, credit score, and activity status, this project estimates the probability of a customer leaving the bank.

The trained ANN model is integrated with a Streamlit application to provide real-time churn predictions.

## Tech Stack
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Streamlit

## Features
- Data preprocessing with label encoding, one-hot encoding, and scaling
- ANN-based churn prediction model
- Real-time prediction through a Streamlit web interface
- Probability-based churn classification

## How It Works
1. User enters customer details in the Streamlit UI  
2. Input data is encoded and scaled using the same preprocessing used during training  
3. The trained ANN model predicts churn probability  
4. The app displays whether the customer is likely to churn or not  

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
