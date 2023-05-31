from flask import Flask, request
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score, precision_recall_fscore_support, accuracy_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

# Import necessary libraries
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

app = Flask(__name__)
pkl_filename = os.getcwd() + "/final_model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

pkl_filename = os.getcwd() + "/scaler.pkl"
with open(pkl_filename, 'rb') as file:
    scaler = pickle.load(file)

example_input = [36,27.4,7.4,2.6,7.6,34.7,0.5,2.1,25.0,1.6,2.3,69.9,0.7,3.4,4.1,1.9,0.4,0.4,1.3]
#[GP, MIN,PTS,GM,FGA,FGP,three_P_Made,three_PA,three_PP,FTM,FTA,FTP,OREB,DREB,REB,AST,STL,BLK,TOV]

@app.route('/search', methods=['GET'])
def search():
    args = request.args
    input = list(args.values())
    input_data = scaler.transform(np.array(example_input).reshape(1,-1))
    prediction_value = model.predict(input_data)
    #prediction_value = model.predict(input_data)
    return {"prediction ": prediction_value.tolist(), "input ": input_data.tolist()}

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8001)

#http://127.0.0.1:8001/search?GP=36&MIN=27.4&PTS=7.4&GM=2.7&FGA=7.6&FGP=34.7&three_P_Made=0.5&three_PA=2.1&three_PP=52&FTM=7&FTA=2&FTP=3&OREB=4&DREB=5&REB=6&ST=7&STL=5&BLK=2&TOV=4
