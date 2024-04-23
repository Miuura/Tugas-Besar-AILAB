import streamlit as st 
import numpy as np
import pandas as pd
import requests

def predict(Area, Room, Parking, Warehouse, Elevator, Address):
    respon = requests.post("http://127.0.0.1:8000/predict", json={"Area" : Area, "Room" : Room, "Parking" : Parking , "Warehouse" : Warehouse, "Elevator" : Elevator, "Address" : Address})

    Hasil = respon.json()["Hasil"]

    return Hasil

df = pd.read_csv('Pricing.csv')
address = df['Address'].unique().tolist()

st.title('Prediction of House Price in Iran')

Area = st.number_input("Enter Area", 10)
Room = st.selectbox('How Many Rooms Do You Need It?', [0,1,2,3,4,5])
Parking = st.selectbox('Do You Need a Parking Area?', ['True','False'])
Warehouse = st.selectbox('Do You Need a WareHouse?', ['True', 'False'])
Elevator = st.selectbox('Do You Need a Elevator?', ['True', 'False'])
Address = st.selectbox('Give me a specific location', address)

if st.button("Predict"):
    Hasil = predict(Area, Room, Parking, Warehouse, Elevator, Address)
    st.write('Prediksi:')
    st.write(f'Harga Dalam Rial Iran: {Hasil:,.2f}')
    st.write(f'Harga Dalam USD: {((Hasil/42000)*15000):,.2f}')