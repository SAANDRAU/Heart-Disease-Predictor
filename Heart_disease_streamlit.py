import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

with open("Heart_Disease.pkl","rb") as file:
    model_data=pickle.load(file)

model=model_data["model"]
Sex_encoded=model_data["Sex_encoded"]
ChestPainType_encoded=model_data["ChestPainType_encoded"]
RestingECG_encoded=model_data["RestingECG_encoded"]
ExerciseAngina_encoded=model_data["ExerciseAngina_encoded"]
ST_Slope_encoded=model_data["ST_Slope_encoded"]
minmax_scaler=model_data["minmax_scaler"]

def main():
    st.title("Heart Disease Prediction Web App")

    Age = st.number_input("Age", min_value=18, max_value=95, value=40, step=1)
    Sex = st.selectbox("Sex",Sex_encoded.classes_)
    ChestPainType = st.selectbox("Chest Pain Type",ChestPainType_encoded.classes_)
    RestingBP = st.number_input("Resting BP",min_value=100,max_value=400,value=100,step=10)
    Cholesterol = st.number_input("Cholesterol",min_value=100,max_value=400,value=100,step=1)
    FastingBS = st.number_input("Fasting BS",min_value=0,max_value=1,value=0,step=1)
    RestingECG = st.selectbox("Resting ECG",RestingECG_encoded.classes_)
    MaxHR = st.number_input("Max HR",min_value=80,max_value=200,value=80,step=1)
    ExerciseAngina = st.selectbox("Exercise Angina",ExerciseAngina_encoded.classes_)
    Oldpeak = st.number_input("Old peak",min_value=0.0,max_value=5.0,value=0.0,step=0.5)
    STSlope = st.selectbox("ST Slope", ST_Slope_encoded.classes_)


    if st.button("Result"):
      input_data=[
         Age,
         Sex_encoded.transform([Sex])[0],
         ChestPainType_encoded.transform([ChestPainType])[0],
         RestingBP,
         Cholesterol,
         FastingBS,
         RestingECG_encoded.transform([RestingECG])[0],
         MaxHR,
         ExerciseAngina_encoded.transform([ExerciseAngina])[0],
         Oldpeak,
         ST_Slope_encoded.transform([STSlope])[0]
        ]
     input_data = np.array(input_data).reshape(1, -1)
     input_data = minmax_scaler.transform(input_data)
     prediction = model.predict(input_data)
     st.success(f"Risk of Heart Disease: {prediction[0]}")
        

if __name__=="__main__":
   main()

    
