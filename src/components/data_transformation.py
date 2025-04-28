import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import saveObject

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifacts","preprocessor.pkl")

class dataTransformation:
    def __init__(self):
        self.dataTransformationConfig = DataTransformationConfig()

    def getDataTransformerObj(self):
        """
        data transformation is for data transformation
        
        """
        try:
            numericalFeatures = ["writing score","reading score"]
            categoricalFeatures = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
            numPipline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median",)),
                    ("Scaler",StandardScaler())
                ]
            )
            logging.info("numerical coloumns standard scalling completed")
            catPipline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("oneHotEncoder",OneHotEncoder()),
                    
                ]
            )
            logging.info("Categorical coloumns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("numerical pipline",numPipline,numericalFeatures),
                    ("Categorical pipline",catPipline,categoricalFeatures)
                ]
            )

            return preprocessor
        except Exception as e: #exception instance (value)
            raise CustomException(e,sys)
    
    def initiateDataTransformatation(self,trainPath,testPath):
        try:
            train_df = pd.read_csv(trainPath)
            test_df = pd.read_csv(testPath)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            prepreocessorOBJ = self.getDataTransformerObj()

            targetColName = "math score"

            df_inputTrainFeatures = train_df.drop(columns=[targetColName],axis=1)
            df_targetFeatureTrain = train_df[targetColName]

            df_inputTestFeatures = test_df.drop(columns=[targetColName],axis=1)
            df_targetFeatureTest = test_df[targetColName]

            logging.info("Applying preprocessing object on training and testing dataframes")

            inputFeatureTrainArrTransformed = prepreocessorOBJ.fit_transform(df_inputTrainFeatures)
            inputFeatureTestArrTransformed = prepreocessorOBJ.transform(df_inputTestFeatures)

            trainArr = np.c_[
                inputFeatureTrainArrTransformed,np.array(df_targetFeatureTrain)
            ]

            testArr = np.c_[
                inputFeatureTestArrTransformed,np.array(df_targetFeatureTest)
            ]
            
            logging.info("Saved preprocessing object")

            saveObject(
                filePath=self.dataTransformationConfig.preprocessor_ob_file_path,
                obj=prepreocessorOBJ
            )

            return(
                trainArr,
                testArr,
                self.dataTransformationConfig.preprocessor_ob_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)