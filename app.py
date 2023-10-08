from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("rf1.pkl", "rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    global abc
    if request.method == "POST":
        
        gender = request.form["gender"]
        if(gender == 'Male'):
            gender_r = 1
        elif(gender == 'Female'):
            gender_r = 0
        else:
            gender_r = 0
        ssc_p = request.form["sscp"]
        sscp = float(ssc_p)
        hsc_p = request.form["hscp"]
        hscp = float(hsc_p)
        deg_p = request.form["degp"]
        degp = float(deg_p)
        etest_p = request.form["etestp"]
        etestp = float(etest_p)
        mba_p = request.form["mbap"]
        mbap = float(mba_p)

        sscb = request.form["sscb"]
        if(sscb == 'Central'):
            ssc_c = 1
            ssc_o = 0
        elif(sscb == 'Others'):
            ssc_c = 0
            ssc_o = 1
        else:
            ssc_c = 0
            ssc_o = 0
        hscb = request.form["hscb"]
        if(hscb == 'Central'):
            hsc_c = 1
            hsc_o = 0
        elif(hscb == 'Others'):
            hsc_c = 0
            hsc_o = 1
        else:
            hsc_c = 0
            hsc_o = 0
    
        hscs = request.form["hscs"]
        if(hscs == 'Arts'):
            hsc_a = 1
            hsc_s = 0
            hsc_c = 0
        elif(hscs == 'Commerce'):
            hsc_a = 0
            hsc_s = 0
            hsc_c = 1
        elif(hscs == 'Science'):
            hsc_a = 0
            hsc_s = 1
            hsc_c = 0
        else:
            hsc_a = 0
            hsc_s = 0
            hsc_c = 0

        degs = request.form["degs"]
        if(degs == 'Commerce & Management'):
            deg_c = 1
            deg_s = 0
            deg_o = 0
        elif(degs == 'Science & Technology'):
            deg_c = 0
            deg_s = 1
            deg_o = 0
        elif(degs == 'Others'):
            deg_c = 0
            deg_s = 0
            deg_o = 1
        else:
            deg_c = 0
            deg_s = 0
            deg_o = 0

        workexperi = request.form["work"]
        if(workexperi == 'Yes'):
            work_exp = 1
        elif(workexperi == 'No'):
            work_exp = 0
        else:
            work_exp = 0
        
        spe = request.form["spe"]

        if(spe == 'Mkt&HR'):
            hr = 1
            fin = 0
        elif(spe == 'Mkt&Fin'):
            hr = 0
            fin = 1
        else:
            hr = 0
            fin = 0

            
        

        query = np.array([gender_r, sscp, hscp, degp, work_exp, etestp, mbap, ssc_c, ssc_o, hsc_c, hsc_o, hsc_a, hsc_c, hsc_s, deg_c, deg_o, deg_s, fin, hr])

        output = model.predict([query])
        
        

        output=round(output[0],2)

        if(output == 1):
            res = 'Placed'
        elif(output == 0):
            res = 'Not Placed'
        else:
            res = 'Invalid Data'

        return render_template('home.html',prediction_text="Result Predicted by Model Is-  {}".format(res))


    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)
