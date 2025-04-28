##Data ingestion = Loading/preparing raw data for your ML pipeline (e.g., reading files, splitting into train/test)

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import dataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_train import ModelTrainer

@dataclass # Automatically generates __init__ and other methods for class variables
class DataIngestionConfig:
    trainDataPath: str=os.path.join("artifacts","train.csv")
    testDataPath: str=os.path.join("artifacts","test.csv")
    rawDataPath: str=os.path.join("artifacts","data.csv")




class DataIngestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()

    def initiateDataIngestion(self):
        logging.info("Enter the data ingestion method or compononet")
        try:
            df = pd.read_csv("../NootBook\dataset\StudentsPerformance.csv")
            logging.info("Read the dataset as data Frame")

            os.makedirs(os.path.dirname(self.ingestionConfig.trainDataPath),exist_ok=True)
            
            df.to_csv(self.ingestionConfig.rawDataPath,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=10)

            pd.DataFrame(train_set).to_csv(self.ingestionConfig.trainDataPath,index=False,header=True)

            test_set.to_csv(self.ingestionConfig.testDataPath,index=False,header=True)

            logging.info("Added dataset to the files")
            logging.info("Ingestion is completed")

            return(
                self.ingestionConfig.trainDataPath,
                self.ingestionConfig.testDataPath,
                self.ingestionConfig.rawDataPath
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    trainData,testData,_ = obj.initiateDataIngestion()

    trainArr,testArr,_ = dataTransformation().initiateDataTransformatation(trainData,testData)
    modelTrainer = ModelTrainer()
    print(modelTrainer.initiatModelTrainer(trainArray=trainArr,TestArray=testArr))