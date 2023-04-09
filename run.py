from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle
import datetime
import pandas as pd


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    result = " "
    if request.method == "POST":
        Zip = int(request.form["Zip"])
        NAICS = int(request.form["NAICS"])
        NoEmp = int(request.form["NoEmp"])
        GrAppv = float(request.form["GrAppv"])
        Term = int(request.form["Term"])
        NewExist = request.form["NewExist"]
        CreateJob = request.form["CreateJob"]
        Franchise = request.form["Franchise"]
        UrbanRural = request.form["UrbanRural"]
        RevLineCr = request.form["RevLineCr"]
        LowDoc = request.form["LowDoc"]
        
        if NAICS != 0:
            NAICS_new = float(str(NAICS)[0:2])
        else:
            NAICS_new = 0
            
        if NewExist == "New":
            NewExist = 2
        else:
            NewExist = 1
            
        if CreateJob == "Yes":
            ISCreateJob = 1
        else:
            ISCreateJob = 0
            
        if Franchise == "Yes":
            ISFranchise = 1
        else:
            ISFranchise = 0
            
        if UrbanRural == "Urban":
            UrbanRural = 1
        elif UrbanRural == "Rural":
            UrbanRural = 2
        else:
            UrbanRural = 0
            
        if RevLineCr == "Yes":
            RevLineCr = 1
        else:
            RevLineCr = 0 

        if LowDoc == "Yes":
            LowDoc = 1
        else:
            LowDoc = 0 

        ZipState = float(str(Zip)[0])
        
        if len(str(Zip)) >= 3:
            ZipCity = float(("".join(list(str(Zip))[1:3])))
        else:
            ZipCity = 0.0  


        
        X = np.array([[Term, NoEmp, NewExist, UrbanRural, RevLineCr, LowDoc, GrAppv, ZipCity, ZipState, NAICS_new, ISCreateJob, ISFranchise]])
        X = pd.DataFrame(X, columns = ['Term', 'NoEmp', 'NewExist', 'UrbanRural', 'RevLineCr', 'LowDoc',
                                       'GrAppv', 'ZipCity', 'ZipState', 'NAICS_new', 'ISCreateJob', 'ISFranchise'])
        result = model.predict_proba(X)
        result = result[:,1]
        result = 'This loan has a default probability of ' + str(round(result[0]*100, 2)) + '%'

        #return render_template("index.html", result=result)
        return render_template("index.html", result=result, form_data=request.form)



if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
