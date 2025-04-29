import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.utils import saveObject,evaluateModels



@dataclass
class modelTrainingConfig:
    tranModelFilePath = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.modelTrainerConfig = modelTrainingConfig()
    

    def initiatModelTrainer(self,trainArray,TestArray):
        try:
            logging.info("Split training and testing data")
            #last column is the target (for both test and training)
            Xtrain,yTrain,Xtest,yTest = (
                trainArray[:,:-1],
                trainArray[:,-1],
                TestArray[:,:-1],
                TestArray[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "decision tree": DecisionTreeRegressor(),
                "gradient boosting":GradientBoostingRegressor(),
                "Linear regression":LinearRegression(),
                "K-Neighbors":KNeighborsRegressor(),
                "adaboost":AdaBoostRegressor(),
                "Support vector regression": SVR()
            }

            params={
                "decision tree":{
                    "criterion":["squared_error","absolute_error"],
                    "splitter":["best","random"],
                    "max_depth":[5,10,15,20,13,14]
                },
                "Linear regression":{

                },
                "Support vector regression":{
                    "kernel":["linear","rbf","poly"]
                } 
            }

            modelReport: dict=evaluateModels(XTrain=Xtrain,yTrain=yTrain,Xtest=Xtest,yTest=yTest,
                                             models=models,params=params)

            bestModelAcc = max(modelReport.values())

            bestmodelName = list(modelReport.keys())[
                list(modelReport.values()).index(bestModelAcc)
                                                     ]
            
            bestModel = models[bestmodelName]
            

            if bestModelAcc<0.6:
                raise CustomException("All models are performing very bad")

            logging.info("Best model found and it is {0} with an accuracy of {1}".format(
                bestmodelName,
                bestModelAcc
            ))

            saveObject(
                filePath=self.modelTrainerConfig.tranModelFilePath,
                obj=bestModel
            )

            predicted = bestModel.predict(Xtest)

            r2Score = r2_score(yTest,predicted)

            return r2Score

        except Exception as e:
            raise CustomException(e,sys)