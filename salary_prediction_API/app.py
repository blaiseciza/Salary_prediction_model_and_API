import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

#Create flask app
app=Flask(__name__)

#chargement du model pickle
model=pickle.load(open("model_salary_data.pkl","rb"))
@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction=model.predict(features)
    return render_template("index.html",prediction_text="Le salaire est de {}".format(prediction))

if __name__=="__main__":
    app.run(debug=True)