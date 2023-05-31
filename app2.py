
# 
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score, precision_recall_fscore_support, accuracy_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

# Import necessary libraries
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

from flask import Flask, jsonify, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

pkl_filename = os.getcwd() + "/final_model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

pkl_filename = os.getcwd() + "/scaler.pkl"
with open(pkl_filename, 'rb') as file:
    scaler = pickle.load(file)

example_input = [36,27.4,7.4,2.6,7.6,34.7,0.5,2.1,25.0,1.6,2.3,69.9,0.7,3.4,4.1,1.9,0.4,0.4,1.3]

#http://127.0.0.1:5000/predict/36/27.4/7.4/2.7/7.6/34.7/0.5/2.1/52/7/2/3/4/5/6/7/5/2/4/
class Prediction(Resource):
    def get(self,GP,MIN,PTS,GM,FGA,FGP,three_P_Made,three_PA,three_PP,FTM,FTA,FTP,OREB,DREB,REB,AST,STL,BLK,TOV):
        print(GP, TOV)
        x = [GP, MIN,PTS,GM,FGA,FGP,three_P_Made,three_PA,three_PP,FTM,FTA,FTP,OREB,DREB,REB,AST,STL,BLK,TOV]
        input_data = scaler.transform(np.array(example_input).reshape(1,-1))
        prediction_value = model.predict(input_data)
        #prediction_value = model.predict(input_data)
        return {"prediction ": prediction_value.tolist()}
        #return {"prediction ": [5]}

api.add_resource(Prediction, '/predict/<GP>/<MIN>/<PTS>/<GM>/<FGA>/<FGP>/<three_P_Made>/<three_PA>/<three_PP>/<FTM>/<FTA>/<FTP>/<OREB>/<DREB>/<REB>/<AST>/<STL>/<BLK>/<TOV>')

if __name__ == "__main__":
    app.run(debug=True,port=8001)

@app.route('/', methods=['POST', 'GET'])
def welcome():
    return "welcome"


"""@app.route('/predict/<GP>/<MIN>/<PTS>/<FGM>/<FGA>/<FGP>/<three_P_Made>/<three_PA>/<three_PP>/<FTM>/<FTA>/<FTP>/<OREB>/<DREB>/<REB>/<AST>/<STL>/<BLK>/<TOV>', methods=['POST', 'GET'])
#GP,MIN,PTS,GM,FGA,FGP,three_P_Made,three_PA,three_PP,FTM,FTA,FTP,OREB,DREB,REB,ST,STL,BLK,TOV
#/<GP><MIN><PTS><FGM><FGA><FGP><three_P_Made><three_PA><three_PP><FTM><FTA><FTP><OREB><DREB><REB><AST><STL><BLK><TOV>
#"http://127.0.0.1:5000/predict/GP=?36,MIN=?27.4,PTS=?7.4,GM=?2.7,FGA=7.6,FGP=34.7,three_P_Made=?0.5,three_PA=?2.1,three_PP=?52,FTM=?7,FTA=?2,FTP=?3,OREB=?4,DREB=?5,REB=?6,ST=?7,STL=?5,BLK=?2,TOV=?4"


#
#
#,MIN,PTS,GM,FGA,FGP,three_P_Made,three_PA,three_PP,FTM,FTA,FTP,OREB,DREB,REB,ST,STL,BLK,TOV
def predict(GP, MIN,PTS,FGM,FGA,FGP,three_P_Made,three_PA,three_PP,FTM,FTA,FTP,OREB,DREB,REB,AST,STL,BLK,TOV):
    print(GP, MIN,PTS,FGM,FGA,FGP,three_P_Made,three_PA,three_PP,FTM,FTA,FTP,OREB,DREB,REB,AST,STL,BLK,TOV)
    pkl_filename = os.getcwd() + "/bayes_brnl.pkl"
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
    pkl_filename = os.getcwd() + "/scaler.pkl"
    with open(pkl_filename, 'rb') as file:
        scaler = pickle.load(file)
    input_data = scaler.transform(np.array([GP, MIN,PTS,FGM,FGA,FGP,three_P_Made,three_PA,three_PP,FTM,FTA,FTP,OREB,DREB,REB,AST,STL,BLK,TOV]).reshape(1,-1))

    prediction = model.predict(input_data) #model.predict(processed_input)[0]    
    # Return the prediction and the image
    return jsonify({'prediction': prediction})
"""
#http://127.0.0.1:5000/predict/GP=?36/MIN=?27.4/PTS=?7.4/GM=?2.7/FGA=?7.6/FGP=?34.7/three_P_Made=?0.5/three_PA=?2.1/three_PP=?52/FTM=?7/FTA=?2/FTP=?3/OREB=?4/DREB=?5/REB=?6/ST=?7/STL=?5/BLK=?2/TOV=?4