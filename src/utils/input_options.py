from conf import SUPPORTED_DATASETS, MAX_FOLDS

input_options = {
    "experiment_type": {
        "type": str, "required": True, "choices": ['standard', 'up_sample']
    },

    "dataset": {
        "type": str, "required": False, "default": SUPPORTED_DATASETS[0], "choices": SUPPORTED_DATASETS,
        "help": "The dataset to train / run the models on."
    },

    "gpus": {
        "type": str, "required": False, "default": "",
        "help": "The gpus to run the models on, use e.g., '0,2' to run on GPU '0' and '2'"
    },

    "nfolds": {
        "type": int, "required": False, "default": MAX_FOLDS, "choices": range(1, MAX_FOLDS + 1),
        "help": "The number of folds to run on."
    },

    "ncores": {"type": int, "required": False, "default": None,
               "help": "The number of cores that each dataloader should use"},

    "nparallel": {
        "type": int, "required": False, "default": 1,
        "help": "The number of processes that should be run on each device"
    },

    "store_best": {
        "type": bool, "required": False, "default": False,
        "help": "Whether the best models found for each run should be stored, "
                "i.e., whether early stopping should be performed."
    },

    "store_every": {
        "type": int, "required": False, "default": 0, "choices": range(0, 100),
        "help": "After which number of epochs the model should be stored, 0 to deactivate this feature"
    },

    "config": {
        "type": str, "required": True,
        "help": "The config file to use when running the model(s)."
    },

    "split": {
        "type": str, "required": False, "default": "test", "choices": ["val", "test"],
        "help": "The split to validate upon."
    },

    "use_tensorboard": {
        "type": bool, "required": False, "default": False,
        "help": "Whether additional information should be logged via tensorboard"
    },

    "experiment": {"type": str, "required": False, "default": None,
                   "help": "The path to an experiment, i.e., collection of multiple runs, "
                           "where each one should be validated"},

    "run": {"type": str, "required": False, "default": None,
            "help": "The path to a run that should be validated."},

    "model_pattern": {
        "type": str, "required": False, "default": "*.pt*",
        "help": "If specified, only models that match this pattern are considered. "
                "(glob syntax is used)"
    }
}