import streamlit as st
import pandas as pd
import joblib

# Load model & encoders
model = joblib.load('Salary_prediction_model .pkl')
encoders = joblib.load('label_encoder_sp.pkl')

st.title('💼 Salary Prediction App')

# Inputs
age = st.number_input('Enter Age', 18, 65, 25)
gender = st.selectbox('Gender', encoders['Gender'].classes_)
education_level = st.selectbox('Education Level', encoders['Education Level'].classes_)
job = st.selectbox('Job Title', encoders['Job Title'].classes_)
experience = st.number_input('Years of Experience', 0.0, 40.0, 2.0)

# Create DataFrame
df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Education Level': [education_level],
    'Job Title': [job],
    'Years of Experience': [experience]
})

# Predict
if st.button("Predict Salary"):
    for col in encoders:
        if col in df.columns:
            df[col] = encoders[col].transform(df[col])
    prediction = model.predict(df)
    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.2f}")
