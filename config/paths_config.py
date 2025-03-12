
import os

CONFIG_PATH="config/config.yaml"
####################### DATA INGESTION #############
RAW_DIR = f"artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")


###################### DATA PROCESSING #############

PROCESSED_DIR = f"artifacts/processed"
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")

###################### MODEL TRAINING #############
MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"

