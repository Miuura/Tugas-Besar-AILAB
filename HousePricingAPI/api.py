from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
app = FastAPI()

# preprocessing data
df = pd.read_csv('Pricing.csv')
df.drop('Price(USD)', axis=1, inplace=True)
df = df.drop_duplicates()
df['Area'] = df['Area'].str.replace(',','')
df['Area'] = df['Area'].astype(float)
df.dropna(inplace=True)

def batasAtas(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR

    return upper

df = df[df['Area'] < batasAtas(df['Area'])]
df = df[df['Price'] < batasAtas(df['Price'])]

df = pd.get_dummies(df, columns=['Address'])
df.replace(True, 1, inplace=True)
df.replace(False, 0, inplace=True)

# split data
X = df.drop(columns = 'Price')
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Model CatBoost
cb = CatBoostRegressor(random_state= 1 , iterations= 1000, learning_rate= 0.1)
cb.fit(X_train, y_train)


df1 = pd.read_csv('Pricing.csv')
df1 = pd.get_dummies(df1, columns=['Address'])
columns = df1.columns
def preprocessing(data):
    data = pd.get_dummies(data, columns=['Address'])
    data = data.replace({'True': 1, 'False': 0})
    data = data.reindex(columns=columns, fill_value=0)
    return data

class PredictionInput(BaseModel):
    Area: float
    Room: int
    Parking: bool
    Warehouse: bool
    Elevator: bool
    Address: str

@app.post("/predict")
def predict(data: PredictionInput):
    data_array = np.array([data.Area, data.Room, data.Parking, data.Warehouse, data.Elevator, data.Address])
    dfData = pd.DataFrame([data_array], columns = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address'])
    predData = preprocessing(dfData)
    
    result = cb.predict(predData)
    rounded_result = round(result[0], 2)

    return {
        "Hasil" : rounded_result
    }

