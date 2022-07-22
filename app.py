import pandas as pd
import streamlit as st 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

st.title('Diabetic Retinopathy Predictor')

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.number_input("Insert the Age")
    systolic_bp = st.sidebar.number_input("Insert Systolic BP")
    diastolic_bp = st.sidebar.number_input("Insert Diastolic BP")
    cholesterol = st.sidebar.number_input("Insert Cholesterol Level")
    data = {'age':age,
            'systolic_bp':systolic_bp,
            'diastolic_bp':diastolic_bp,
            'cholesterol':cholesterol}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

data = pd.read_csv("pronostico_dataset.csv",sep=';')
data=data.drop('ID',axis=1)
data = data.replace({'prognosis': {'retinopathy': 1, 
                                'no_retinopathy': 0}})
                                
X = data.drop('prognosis',axis=1)
y = data["prognosis"]

classifier =  SVC(kernel='rbf', C=2, probability=True)
classifier.fit(X,y)
y_pred = classifier.predict(df)
prediction_proba = classifier.predict_proba(df)


st.subheader('Predicted Result')
st.write('You might be suffering from Diabetic Retinopathy' if prediction_proba[0][1] > 0.5 else 'You are NOT suffering from Diabetic Retinopathy')