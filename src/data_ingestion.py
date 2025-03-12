import kagglehub
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from config.paths_config import *
from utils.custom_functions import read_yaml
from src.logger import get_logger
from src.custom_exception import CustomException


CONFIG_FILE_NAME=f"config/config.yaml"

logger = get_logger(__name__)
class DataIngestion:
    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_NAME)
        self.data_path = kagglehub.dataset_download(self.config['data_ingestion']['kaggle_file_name'])
        self.raw_dir = RAW_DIR
        self.raw_data_path = RAW_FILE_PATH
        self.train_data_path = TRAIN_FILE_PATH
        self.test_data_path = TEST_FILE_PATH

    def load_raw_data(self, ):
        try:
            logger.info(f'Starting data loading')
            os.makedirs(self.raw_dir, exist_ok = True)
            df=pd.read_csv(os.path.join(self.data_path,self.config['data_ingestion']['file_name']))

            logger.info(f'Raw data saved successfully at {self.raw_data_path}')
            return df

        except Exception as e:
            logger.error(f'Error in loading data {e}')
            raise CustomException('Error while loading data', e)


    def split_and_save_data(self, df):
        try:
            logger.info(f'Starting data split')
            train_df, test_df= train_test_split(df, test_size = 1 - self.config['data_ingestion']['train_ratio'])
            
            logger.info(f'Data split successful')
            df.to_csv(self.raw_data_path)
            train_df.to_csv(self.train_data_path)
            test_df.to_csv(self.test_data_path)

            logger.info(f'Data saved successfully at {self.train_data_path} and {self.test_data_path}')

        except Exception as e:
            logger.error(f'Error in loading data {e}')
            raise CustomException('Error while loading data', e)
        
    def run(self):
        try:
            logger.info('Starting data ingestion')
            
            df = self.load_raw_data()
            self.split_and_save_data(df)

            logger.info('Data ingestion successful')

        except Exception as e:
            logger.error(f'Error in Ingesting data {e}')
            raise CustomException('Error during data ingestion', e)


if __name__=='__main__':
    data_ingestion = DataIngestion()
    data_ingestion.run()
    
