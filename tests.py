import requests
import pandas as pd
data = pd.read_csv('data.txt', sep = ';') 
data.drop(data.index[data['DATA_TYPE'] == 'TRAIN'], inplace = True)
del data['DATA_TYPE']
data = data.to_dict('index')
for example in data:
    print(data[example])
    print(requests.post('http://127.0.0.1:5000/api/predict', json = data[example]).text)
