import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            params = yaml.safe_load(open("params.yaml"))['filter']
            df = pd.read_csv('data\insurance.csv')

            df = df[df['age'] <= params['age_value']]
            print(df)

            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=23)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    transformer = DataTransformation()
    train_arr, test_arr,_ = transformer.initiate_data_transformation(train_data, test_data)

    trainer = ModelTrainer()
    best_model_name, r2_sc = trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Model used : {best_model_name}")
    print(f"R2_score  : {r2_sc}")