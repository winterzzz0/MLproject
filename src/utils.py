#utils.py = Shared helper functions/classes ()
import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
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
    

def evaluateModels(XTrain,yTrain,Xtest,yTest,models:dict,params)->dict:
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            modelName = list(models.keys())[i]
            param = params.get(modelName)
            if param:
                gridSeach = GridSearchCV(estimator=model,param_grid=param,cv=4)
                gridSeach.fit(XTrain,yTrain)
                models[modelName] = gridSeach.best_estimator_
                model = models[modelName]
            else:
                model.fit(XTrain,yTrain)
            
           

            yTrainPred = model.predict(XTrain)

            yTestPred = model.predict(Xtest)

            trainModelR2 = r2_score(yTrain,yTrainPred)

            testModelR2 = r2_score(yTest,yTestPred)

            report[list(models.keys())[i]]= testModelR2

        return report
        
    except Exception as e:
        raise CustomException(e,sys)


def loadObj(filePath):
    try:
        with open(filePath,"rb") as fileObj:
            return dill.load(fileObj)
    except Exception as e:
        raise CustomException(e,sys)