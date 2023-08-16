import os, sys

import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

            models ={
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "K-nearest Regressor": KNeighborsRegressor()
            }

            params ={
                "Linear Regression": {},

                "Decision Tree": {
                    'criterion': ['squared_error','friedman_mse','absolute_error'],
                    'max_depth': [3,4,5,6,8]
                    },

                "Random Forest": {
                    'criterion': ['squared_error','friedman_mse','absolute_error'],
                    'max_depth': [3,4,5,6,8]
                    },

                "Gradient Boost": {
                    'learning_rate': [0.1, 0.05, 0.5, 0.01]
                    },

                "AdaBoost": {
                    'learning_rate': [0.1, 0.05, 0.5, 0.01]
                    },

                "K-nearest Regressor": {
                    'n_neighbors': [3,5,7,9]
                    }
            }

            
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found on both train and test dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            ) 

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return (best_model_name , r2_square)


        except Exception as e:
            raise CustomException(e, sys)