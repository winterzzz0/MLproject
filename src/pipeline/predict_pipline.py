import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import loadObj



class PredictPipline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            modelPath = r"D:\Cources\End-to-End ML project\src\components\artifacts\model.pkl"
            preprocessorPath = r"D:\Cources\End-to-End ML project\src\components\artifacts\preprocessor.pkl"
            model = loadObj(modelPath)
            preprocessor = loadObj(preprocessorPath)
            scaledData = preprocessor.transform(features)
            predictions = model.predict(scaledData)
            return predictions
        except Exception as e:
             raise CustomException(e,sys)
         
              
            


class CustomData:
    def __init__(self,
                gender:str,
                race_ethnicity: str,
                parental_level_of_education: str,
                lunch:str,
                test_preparation_course:str,
                reading_score:float,
                writing_score:float):
                self.gender = gender
                self.race_ethnicity = race_ethnicity
                self.parental_level_of_education = parental_level_of_education
                self.lunch = lunch
                self.test_preparation_course = test_preparation_course
                self.reading_score = reading_score
                self.writing_score = writing_score
    
    def getDataAsFrame(self):
          try:
                CustomDataInputDict = {
                      "gender": [self.gender],
                      "race/ethnicity": [self.race_ethnicity],
                      "parental level of education": [self.parental_level_of_education],
                      "lunch": [self.lunch],
                      "test preparation course": [self.test_preparation_course],
                      "reading score": [self.reading_score],
                      "writing score": [self.writing_score]
                }
                return pd.DataFrame(CustomDataInputDict)

          except Exception as e:
            raise CustomException(e,sys)
