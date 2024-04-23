import streamlit as st
import numpy as np
import pandas as pd
import joblib

df = pd.read_csv('Pricing.csv')
address = df['Address'].unique().tolist()
df = pd.get_dummies(df, columns=['Address'])
columns = df.columns

model = joblib.load('catboost_moodel.joblib')

def preprocessing(data):
    data = pd.get_dummies(data, columns=['Address'])
    data = data.replace({'True': 1, 'False': 0})
    data = data.reindex(columns=columns, fill_value=0)
    return data

st.title('Prediction of House Price in Iran')

Area = st.number_input("Enter Area", 10)
Room = st.selectbox('How Many Rooms Do You Need It?', [0,1,2,3,4,5])
Parking = st.selectbox('Do You Need a Parking Area?', ['True','False'])
Warehouse = st.selectbox('Do You Need a WareHouse?', ['True', 'False'])
Elevator = st.selectbox('Do You Need a Elevator?', ['True', 'False'])
Address = st.selectbox('Give me a specific location', address)

def predictPrice(Area, Room, Parking, Warehouse, Elevator, Address):
    data = np.array([Area, Room, Parking, Warehouse, Elevator, Address])
    dfData = pd.DataFrame([data], columns = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address'])
    predData = preprocessing(dfData)
    
    result = model.predict(predData)[0]
    rounded_result = round(result, 2)

    st.write('Perkiraan Harga Rumah: ')
    st.write(f'Harga Dalam Riel: {rounded_result:,.2f}')
    st.write(f'Harga Dalam USD: {(rounded_result/42000):,.2f}')
   
if st.button("Predict"):
    predictPrice(Area, Room, Parking, Warehouse, Elevator, Address)