import multiprocessing

SUPPORTED_DATASETS = ["lfm2b", "movielens"]
MAX_FOLDS = 5

# Thresholds for evaluation
LEVELS = (
    10, 20, 50
)

# Which demographic trait to consider
DEMO_TRAITS = (
    "gender",
)

EXP_SEED = 101315  # Seed for algorithms that rely on random initialization

# Parameters for VAE
# instead of adjusting this value, consider to specify {"general_params": {"n_epochs": <some_value>}}
# in the config file you are using instead
VAE_MAX_EPOCHS = 50

VAE_LOG_VAL_EVERY = 1
VAE_LOG_VAL_METRICS_EVERY = 3  # metric calculations take a lot of time (as they have to be done per user)

TR_LOG_IN_BETWEEN_EPOCH_EVERY = 30

N_WORKERS = min(10, multiprocessing.cpu_count())  # prevent excessive use of workers if no more cpu cores are available
BATCH_SIZE = 64

VAL_METRICS = ("ndcg", "recall")
VAL_LEVELS = (10, 20, 50)

# instead of adjusting this value, consider to specify {"atk_params": {"n_epochs": <some_value>}}
# in the config file you are using instead
ATK_MAX_EPOCHS = 25

DATA_PATH = "<your_data_path>/sampled_100000_items_inter.txt"
DEMO_PATH = "<your_data_path>/sampled_100000_items_demo.txt"
TRACKS_PATH = "<your_data_path>/sampled_100000_items_tracks.txt"

OUT_DIR = "<your_data_path>/data/{}/"

MOVIELENS_PATH = "<your_data_path>/user_gte_5_movie_gte_5"

# Path for storing results, parameters are {experiment_type} and {time_stamp}
LOG_DIR = "./../results/<dataset>/{}--{}/"

ACCEPTABLE_MODEL_DIRS = ("vae", "retrain*")

