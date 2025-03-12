from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelConfig
from config.paths_config import *

if __name__=="__main__":

    data_ingestion = DataIngestion()
    data_ingestion.run()

    processor=DataProcessor(
        train_path = TRAIN_FILE_PATH,
        test_path = TEST_FILE_PATH, 
        processed_dir = PROCESSED_DIR, 
        config_path = CONFIG_PATH
        )
    processor.process()

    trainer = ModelConfig(PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, MODEL_OUTPUT_PATH)
    trainer.run()

