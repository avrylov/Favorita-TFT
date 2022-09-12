import os
from decouple import config

# path settings
ABS_PROJECT_PATH = config('ABS_PROJECT_PATH')
PROC_DATA_PATH = os.path.join(ABS_PROJECT_PATH, 'data/processed')
SUBMISSION_PATH = os.path.join(ABS_PROJECT_PATH, 'data/submissions')
CHECK_POINT_PATH = os.path.join(ABS_PROJECT_PATH, 'data/check_points')
LOGS_PATH = os.path.join(ABS_PROJECT_PATH, 'data/logs')

# tft dataset params
MAX_PREDICTION_LENGTH = 16  # how many days to predict in test
MAX_ENCODING_LENGTH = 90
BATCH_SIZE = 128
NUM_WORKERS = 4

# tft model params
MAX_EPOCHS = 150
LIMIT_TRAIN_BATCHES = 1000
LSTM_LAYERS = 2
OUTPUT_SIZE = 7
REDUCE_ON_PLATEAU_PATIENCE = 2

# tft tuninig params
N_TRIALS = 5
MAX_EPOCHS_TUNE = 40

