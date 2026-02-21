# Importing Libraries

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf

#Load Trained Model
model = tf.keras.models.load_model('model.h5')


#Load One Hot Encoded file of Geography

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)


#Load Label Encoded file of Gender

with open ('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

#Loading the Scaler File

with open('scaler.pkl','rb') as file:
    Scaler = pickle.load(file)

st.title("Customer Churn Prediction")


#Taking User input Only

geography= st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender= st.selectbox('Gender',label_encoder_gender.classes_)
age= st.slider('Age',18,99)
balance= st.number_input('Balance')
credit_score= st.number_input('Credit Score')
estimated_salary= st.number_input('Estimated Salary')
tenure= st.slider('Tenure',0,10)
num_of_products= st.slider("Number of Products",1,4)
has_cr_card= st.selectbox("Has Credit Card",[0,1])
is_active_member=st.selectbox("Is Active Member",[0,1])

# We have to this Data in Input so it can be used

###### Mapping Input Data with our Machine

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

#As we can see in the above we haven't mapped Geography data, for that we have to do seprate step and then conacte it

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#Concate/Merge the Geography data with Input data

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


#Scale all the input Data
input_data_scaled = Scaler.transform(input_data)

#Predict Churn

prediction = model.predict(input_data_scaled)
predict_proba=prediction[0][0]   #probability

st.write(f"Churn Probabilty: {predict_proba: .2f}")  #2f mean prediction can be 2 decimals

if predict_proba>0.5:
    st.write("The Customer is likely to leave Bank.")
else:
    st.write("The Customer is not likely to leave the Bank.")