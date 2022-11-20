import pandas as pd
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')

print("the first 5 rows of our data looks like this :\n")
print(concrete_data.head())

print("Let's check if there are any mising values :\n")
print(concrete_data.isnull().sum())

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] 
target = concrete_data['Strength']

predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

n_cols = predictors_norm.shape[1]



import keras
from keras.models import Sequential
from keras.layers import Dense

def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = regression_model()
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
