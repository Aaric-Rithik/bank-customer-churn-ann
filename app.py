import tensorflow as tf
import pickle
import pandas as pd
import streamlit as st

# Load the trained model
model = tf.keras.models.load_model("model_ann_bank_churn.h5")

# Load encoders and scaler
with open("one_hot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Bank Customer Churn Prediction")

# User input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92, 40)
balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600, step=1)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)
tenure = st.slider("Tenure", 0, 10, 3)
num_of_products = st.slider("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1], index=1)
is_active_member = st.selectbox("Is Active Member", [0, 1], index=1)

# Build input dataframe
input_df = pd.DataFrame([{
    "CreditScore": credit_score,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
    "Geography": geography
}])

# Encode Geography (one-hot) - IMPORTANT: convert sparse to array
geo_encoded = onehot_encoder_geo.transform(input_df[["Geography"]])
if hasattr(geo_encoded, "toarray"):
    geo_encoded = geo_encoded.toarray()

geo_cols = onehot_encoder_geo.get_feature_names_out(["Geography"])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_cols[: geo_encoded.shape[1]])

# Encode Gender (label encoding)
input_df["Gender"] = label_encoder_gender.transform(input_df["Gender"])

# Drop original Geography and add encoded columns
input_df = pd.concat([input_df.drop(columns=["Geography"]), geo_encoded_df], axis=1)

# Ensure column order matches training scaler (if available)
if hasattr(scaler, "feature_names_in_"):
    input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)
prediction_proba = float(prediction[0][0])


st.write(f"Churn probability: **{prediction_proba:.4f}**")

if prediction_proba > 0.5:
    st.error("⚠️ Customer is likely to churn")
else:
    st.success("✅ Customer is not likely to churn")
