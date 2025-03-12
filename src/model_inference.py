import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read yaml, load_data
from scipy.stats import randint
import mlflow

logger = get_logger(__name__)

class ModelConfig:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path =train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_list = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
    
    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)
            
            X_train = train_df.drop(columns = 'booking_status')
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns = 'booking_status')
            y_test = test_df['booking_status']
            
            logger.info(f"Loading data from {self.train_path}")
            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed during load data", e)
        
    def train_lightgbm(self, X_train, y_train):
        try:
            logger.info(f"Initializaing model")
            
            lgbm_model = lgb.LGBMClassifier(random_state = self.random_search_params['random_state'])
            
            logger.info(f"Starting hyper-parameter tuning")

            random_search = RandomizedSearchCV(
                estimator = lgbm_model,
                param_distributions= self.params_dict,
                n_iter= self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring = self.random_search_params['scoring']
            )

            random_search.fit(X_train,y_train)

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best parameters are {best_params}")

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error while model training {e}")
            raise CustomException("Failed during model training", e)
        
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info(f"Staring evaluation")
            y_pred = model(X_test)
            
            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
            precision = precision_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred)
            f1 = f1_score(y_true=y_test, y_pred=y_pred)
            
            logger.info(f"Accuracy : {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall : {recall}")
            logger.info(f"F1 score : {f1}")

            logger.info(f"Evaluation Complete")

            return {
                "Accuracy" : accuracy,
                "Precision" : precision,
                "Recall" : recall,
                "F1 score" : f1
            }
        
        except Exception as e:
            logger.error(f"Error while model evaluation {e}")
            raise CustomException("Failed to evaluate model", e)
    
    def save_model(self, model):
        try:
            logger.info(f"Saving model")

            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.save(model, self.model_output_path)

            logger.info(f"Model saved to {self.model_output_path}")


        
        except Exception as e:
            logger.error(f"Error while model saving {e}")
            raise CustomException("Failed to save model", e)
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info(f"Staring Model training pipeline")
                logger.info(f"Starting ML flow experiment tracking")

                X_train, y_train, X_test, y_test = self.load_and_split_data()

                mlflow.log_artifact()
                best_lightgbm_model = self.train_lightgbm(X_train, y_train)
                metric = self.evaluate_model(best_lightgbm_model, X_test,y_test)
                self.save_model(best_lightgbm_model)

                mlflow.log_artifact('artifacts')
                logger.info(f"Model training successful")
        
        except Exception as e:
            logger.error(f"Error in model training pipeline {e}")
            raise CustomException("Failed during model training pipeline", e)

if __name__ == '__main__':
    trainer = ModelConfig(PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, MODEL_OUTPUT_PATH)
    trainer.run()







