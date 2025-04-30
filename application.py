from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipline import CustomData,PredictPipline

application = Flask(__name__)

@application.route("/")
def index():
    return render_template("index.html")

@application.route("/oredictData",methods=["GET","POST"])
def predictDataPoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        
        data=CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=request.form.get("reading_score"),
            writing_score=request.form.get("writing_score")

        )
        pred_df = data.getDataAsFrame()
        print(pred_df)
        predPipline = PredictPipline()
        results = predPipline.predict(pred_df)
        return render_template("home.html",res=results[0])
    
if __name__ == "__main__":
    application.run(host="0.0.0.0",debug=True)