#utils.py = Shared helper functions/classes ()
import os
import sys
import dill
import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.metrics import r2_score

def saveObject(filePath,obj):
    try:
        dirPath = os.path.dirname(filePath)
        os.makedirs(dirPath,exist_ok=True)
        with open(filePath,"wb") as fileObj:
            dill.dump(obj,fileObj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluateModels(XTrain,yTrain,Xtest,yTest,models)->dict:
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i] #keys

            model.fit(XTrain,yTrain)

            yTrainPred = model.predict(XTrain)

            yTestPred = model.predict(Xtest)

            trainModelR2 = r2_score(yTrain,yTrainPred)

            testModelR2 = r2_score(yTest,yTestPred)

            report[list(models.keys())[i]]= testModelR2

        return report
        
    except Exception as e:
        raise CustomException(e,sys)