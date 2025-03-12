import os
from yaml import safe_load 
from src.logger import get_logger
from src.custom_exception import CustomException

logger=get_logger(__name__)


def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File is not at given path')
        
        print(file_path)
        
        with open(file_path, "r") as f:
            config=safe_load(f)
            logger.info(f"File {file_path} read successful")
        return config
    
    except Exception as e:
        logger.error("Data load error")
        raise CustomException(f"Error while file {file_path}", e)

# if __name__=='__main__':
#     read_yaml('config/config.yaml')


    