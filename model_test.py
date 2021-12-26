# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 15:05:21 2021

@author: akhil
"""

import joblib
from sklearn.metrics import f1_score
import pandas as pd

df  = pd.read_csv("./encounter_data.csv")

#preprocessing the data
from sklearn.preprocessing import MinMaxScaler
std = MinMaxScaler()
df[["lat","lon","vessel.mmsi","median_speed_knots","elevation_m","distance_from_shore_m","distance_from_port_m"]] = std.fit_transform(df[["lat","lon","vessel.mmsi","median_speed_knots","elevation_m","distance_from_shore_m","distance_from_port_m"]])

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
labels = ["start","id","end","vessel.id","vessel.type","vessel.flag","vessel.name","regions.rfmo"]

def encode_data():
    for label in labels:
        df[label] = encoder.fit_transform(df[label])

encode_data()


X = df.drop(columns=["type"], axis=1)
y = df[["type"]]

#splitting data
from sklearn.model_selection import train_test_split
import sklearn

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

def test_f1score():
    model = joblib.load("./finalized_model.sav")
    
    predictions = model.predict(X_test)
    
    score = f1_score(predictions, y_test, average="weighted")
    
    if score > 0.9:
        score == 1
    else:
        score == 0
    
    assert score == 1
    