import os
import shutil
import multiprocessing

from tqdm import trange
from joblib import Parallel, delayed
from collections import OrderedDict
from sklearn.model_selection import ParameterGrid

from src.modules.multi_vae import MultiVAEAdv
from src.utils.input_validation import parse_input
from src.utils.vae_attack_utils import attack_single
from src.utils.vae_training_utils import train_single
from src.utils.nn_unils import get_datasets_and_loaders
from src.utils.helper import reproducible, replace_special_path_chars, load_config, prepare_config, get_log_dir
from conf import EXP_SEED, DEMO_TRAITS, VAE_LOG_VAL_EVERY, \
    TR_LOG_IN_BETWEEN_EPOCH_EVERY, VAE_LOG_VAL_METRICS_EVERY, \
    VAL_METRICS, VAL_LEVELS

"""
About:
We want to train an multi-variational auto encoder (MultiVAE adv) to generate new recommendations for a
user base while reducing the bias that a model may learn due to imbalanced / incomplete datasets.

In order to produce meaningful results, we will upsample the dataset 
for training the attacker network, upsampled
data will be used. Otherwise, the classifier may just learn to predict the majority
class as we are working with imbalanced datasets.

Further, we train an attacker network for the best model on each configuration 
"""

if __name__ == '__main__':
    input_config = parse_input("experiments", options=["experiment_type", "nfolds", "ncores", "nparallel", "gpus",
                                                       "config", "store_best", "store_every", "dataset"])

    upsample_data = True
    fixed_params, grid_search_params = load_config(input_config.config)

    if grid_search_params.get("adv|in_use") is not None:
        raise AttributeError("Training with / without adversaries must be fixed (due to data loading)! \n"
                             "Please remove it from the grid search.")

    # use adversarial network is optional and defaults to False
    use_adv_network = bool(fixed_params.get("adv|in_use"))

    # user may not even want to run a grid search, enable possibility
    # to use fixed params only
    perform_grid_search = len(grid_search_params) > 0
    if perform_grid_search:
        pg = ParameterGrid(grid_search_params)
    else:
        pg = [fixed_params]
        fixed_params = dict()

    log_base_str = get_log_dir(input_config.dataset, input_config.experiment_type)
    base_folder = os.path.dirname(log_base_str)

    # Copy config file used for the parameter search to easier compare different runs
    os.makedirs(base_folder, exist_ok=True)
    shutil.copyfile(input_config.config, os.path.join(base_folder, "used_config.json"))

    for fold_n in trange(input_config.nfolds, desc='folds'):
        log_str = os.path.join(log_base_str, str(fold_n))

        if use_adv_network and not upsample_data:
            raise AttributeError("For adversarial networks up-sampled data has to be used!")

        # Setting seed for reproducibility
        reproducible(EXP_SEED)

        # Load data
        dataset_and_loaders = get_datasets_and_loaders(dataset_name=input_config.dataset, fold=fold_n,
                                                       splits=("train", "val", "test"), n_workers=input_config.ncores,
                                                       run_parallel=True, oversample_train=upsample_data)
        tr_set, tr_loader = dataset_and_loaders["train"]
        vd_set, vd_loader = dataset_and_loaders["val"]
        te_set, te_loader = dataset_and_loaders["test"]

        print(f"train set contains {len(tr_set)} samples")
        print(f"validation set contains {len(vd_set)} samples")
        print(f"test set contains {len(te_set)} samples")
        print(f"total samples: {len(tr_set) + len(vd_set) + len(te_set)}")
        print("Data Loaded")

        devices = input_config.devices
        n_devices = len(devices)


        # wrap the training process such that we can distribute training on multiple gpus
        def process(pg_config, verbose=False):
            identity = multiprocessing.current_process()._identity
            process_id = 0 if len(identity) == 0 else identity[0]
            print(f"Process {process_id} running with config is {pg_config}")
            device = devices[process_id % n_devices]
            config_str = replace_special_path_chars(str(pg_config)) if perform_grid_search else "run"
            config = prepare_config(pg_config, fixed_params, tr_set.n_items)

            _, best_model_dicts = train_single(log_dir=log_str, run_name=config_str, config=config,
                                               use_adv_network=use_adv_network, device=device,
                                               user_groups_all_traits=tr_set.user_groups_per_trait, tr_loader=tr_loader,
                                               vd_loader=vd_loader,
                                               log_batch_results_every=TR_LOG_IN_BETWEEN_EPOCH_EVERY,
                                               log_val_every=VAE_LOG_VAL_EVERY,
                                               log_val_metrics_every=VAE_LOG_VAL_METRICS_EVERY, traits=DEMO_TRAITS,
                                               val_levels=VAL_LEVELS, val_metrics=VAL_METRICS, verbose=verbose,
                                               store_best_model=input_config.store_best,
                                               store_model_every=input_config.store_every)

            for metric, sd in best_model_dicts.items():
                if verbose:
                    print(f"\nLoading pretrained model for metric '{metric}'")
                # Load pretrained model that we want to evaluate
                pretrained_model = MultiVAEAdv(**config["model"], adv_config=None)  # no need for adversaries

                used_keys = pretrained_model.state_dict().keys()
                pretrained_model.load_state_dict(OrderedDict([(k, v) for k, v in sd.items() if k in used_keys]))

                pretrained_model.to(device)
                pretrained_model.eval()

                run_name = config_str + "_" + metric
                attack_single(log_dir=log_str, run_name=run_name,
                              pretrained_model=pretrained_model,
                              attacker_config=config["atk"], attacker_opt=config["atk_opt"],
                              tr_loader=tr_loader, vd_loader=vd_loader, te_loader=te_loader,
                              device=device, log_batch_results_every=TR_LOG_IN_BETWEEN_EPOCH_EVERY,
                              verbose=verbose)


        print("\n")
        print("=" * 60)
        print(f"Starting training for {len(pg)} configuration(s)")
        print("=" * 60, "\n")
        # Running hyperparameter search
        Parallel(n_jobs=n_devices, verbose=11)(
            delayed(process)(c, verbose=not perform_grid_search) for c in pg)
