import os
import pandas as pd
import numpy as np

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.custom_functions import read_yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path=train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)
        os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess_data(self, df):
        try:
            logger.info("Starting data processing")

            logger.info("Dropping the columns")
            df.drop(columns = 'Booking_ID', inplace=True)

            logger.info("Dropping the duplicates")
            df.drop_duplicates(inplace=True)
            cat_cols = self.config["data_processing"]['categorical_cols']
            num_cols = self.config["data_processing"]['numerical_cols']

            logger.info("Applying label_encoding")
            label_encoder=LabelEncoder()
            mappings={}
            for col in cat_cols:
                df[col]=label_encoder.fit_transform(df[col])
                mappings[col] = {label: code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

            logger.info("Mappings are: ")
            for label, code in mappings.items():
                logger.info(f'{label} : {code}')

            logger.info('Doing skewness handling')

            skewness= df.skew()
            for col in df.columns:
                if skewness[col]> self.config["data_processing"]['skewness_threshold']:
                    df[col] = np.log1p(df[col])
            
            return df
        
        except Exception as e:
            logger.error(f"Error during preprocessing step {e}")
            raise CustomException(f"Error while preprocessing data", e)

    def balance_data(self, df):
        try:
            logger.info("Handling imbalance data")

            X=df.drop(columns='booking_status')
            y=df['booking_status']

            smote=SMOTE(random_state=24)
            X_resampled, y_resampled = smote.fit_resample(X,y)

            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['booking_status'] = y_resampled

            logger.info('Data balanced successfully')
            return balanced_df

        except Exception as e:
            logger.error(f"Error during balancing data {e}")
            raise CustomException(f"Error while balancing data", e)

    def feature_selection(self, df):
        try: 
            logger.info('Starting our feature selection step')

            X=df.drop(columns='booking_status')
            y=df['booking_status']

            model = RandomForestClassifier(random_state=22)
            model.fit(X,y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance' : feature_importance
                })
            
            top_features_important = feature_importance_df.sort_values(by = "importance", ascending=False)
            num_features_to_select = self.config["data_processing"]["no_of_features"]

            logger.info(f"number of features selected {num_features_to_select}")

            top_10_features = top_features_important["feature"].head(num_features_to_select).values
            top_10_df = df[top_10_features.tolist()+['booking_status']]


            logger.info("Successfully selected features")
            return top_10_df
        
        except Exception as e:
            logger.error(f"Error during feature selection {e}")
            raise CustomException(f"Error while feature selection", e)

    def save_data(self, df, save_path):
        try: 
            logger.info('Starting data saving')

            df.to_csv(save_path)

            logger.info(f"data saved successfully to {save_path}")
        
        except Exception as e:

            logger.error(f"Error during data saving {e}")
            raise CustomException(f"Error while saving data", e)
    
    def process(self,):
        try:
            logger.info("Loading data from RAW directory")
            train_df = pd.read_csv(self.train_path)
            test_df =pd.read_csv(self.test_path)

            train_df = self.preprocess_data(train_df)
            train_df = self.balance_data(train_df)

            test_df = self.preprocess_data(test_df)

            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_PATH)
            self.save_data(test_df, PROCESSED_TEST_PATH)

            logger.info("Data processing completed successfully")

        except Exception as e:
            logger.error(f"Error during data saving {e}")
            raise CustomException(f"Error while saving data", e)

if __name__=="__main__":
    processor=DataProcessor(
        train_path = TRAIN_FILE_PATH,
        test_path = TEST_FILE_PATH, 
        processed_dir = PROCESSED_DIR, 
        config_path = CONFIG_PATH
        )
    processor.process()





