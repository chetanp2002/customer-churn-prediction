import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# load the trained model 
model= tf.keras.models.load_model("models/model.h5")

# load the encoder and scaler 
with open('models/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender= pickle.load(file)

with open('models/onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo= pickle.load(file)

with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title("Customer Churn Prediction")

# User input 
geography= st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender= st.selectbox('Gender', label_encoder_gender.classes_)
age= st.slider('Age', min_value=18, max_value=100)
balance= st.number_input('Balance')
credit_score= st.number_input("Credit Score")
estimated_salary= st.number_input('Estimated Salary')
tenure= st.slider('Tenure', 0, 10)
num_of_preducts= st.slider('Numcer of Products',1,4)
has_cr_card= st.selectbox("Has Credit Card",[0,1])
is_active_member= st.selectbox("Is Active Member", [0,1])

# Prepare the input data 
input_data= pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_preducts],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One hot encode 'Geography'
geo_encoded= onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df= pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# combine one hot encoded columns with input data 
input_data= pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scale the data using the scaler
input_data_scaled= scaler.transform(input_data)

# Prediction churn
prediction= model.predict(input_data_scaled)
prediction_proba= prediction[0][0]

# st.write(f"Churn Probability: {prediction_proba:.2f}")

# if prediction_proba > 0.5:
#     st.write("The customer is likely to churn.")
# else:
#     st.write("The customer is unlikely to churn.")

# Display the churn probability
st.write(f"Churn Probability: {prediction_proba:.2f}")

# Stylish outcome display
if prediction_proba > 0.5:
    # Use a red color and a warning icon for high churn probability
    st.markdown(
        "<h3 style='color: red; text-align: center;'>🚨 The customer is likely to churn 🚨</h3>",
        unsafe_allow_html=True
    )
else:
    # Use a green color and a thumbs-up emoji for low churn probability
    st.markdown(
        "<h3 style='color: green; text-align: center;'>👍 The customer is unlikely to churn 👍</h3>",
        unsafe_allow_html=True
    )