import os
import torch
from tqdm import trange

import multiprocessing
from joblib import Parallel, delayed

from src.modules.multi_vae import MultiVAEAdv
from src.utils.vae_attack_utils import attack_single
from src.utils.nn_unils import get_datasets_and_loaders, checkout_run_dict, run_to_fold_dict
from src.utils.input_validation import parse_input
from src.utils.helper import reproducible, json_load, load_config_eval

from conf import EXP_SEED, TR_LOG_IN_BETWEEN_EPOCH_EVERY

"""
About:
We want to train an attacker network on the latent space of an variational autoencoder (more precisely, 
a MultiVAE with an adversarial network), to predict demographic features such as gender of the users.

In order to produce meaningful results, for training the attacker network, upsampled
data will be used. Otherwise, the classifier may just learn to predict the majority
class as we are working with imbalanced datasets.

Note that as we simply want to determine the remaining information in the latent space, 
we don't really need, and therefore don't support a grid search over the attackers' hyperparemeters.
"""

if __name__ == '__main__':
    input_config = parse_input("attack", options=["run", "experiment", "nfolds", "ncores", "nparallel", "gpus",
                                                  "split", "use_tensorboard", "config"])

    run_dict = input_config.run_dict
    if not checkout_run_dict(run_dict):
        exit()
    fold_dict = run_to_fold_dict(run_dict)

    attacker_config = json_load(input_config.config)

    for fold_n in trange(input_config.nfolds, desc='folds'):

        # Setting seed for reproducibility
        reproducible(EXP_SEED)

        # Load data
        dataset_and_loaders = get_datasets_and_loaders(dataset_name=input_config.dataset, fold=fold_n,
                                                       splits=("train", "val", "test"), n_workers=input_config.ncores,
                                                       run_parallel=True, oversample_train=True)
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


        # wrap the training process into another function such that we can
        # distribute training on multiple gpus
        def process(run_dir, run_data, verbose=False):
            identity = multiprocessing.current_process()._identity
            process_id = 0 if len(identity) == 0 else identity[0]
            print(f"Process {process_id} running with config to attack ")
            device = devices[process_id % n_devices]

            config, use_adv_network = load_config_eval(run_data["config_file"])
            for model_name, model_path in run_data["models"]:
                print(f"\nRunning validation for \n{run_dir} \nand model \n{model_name}")

                # Although we won't use the adversarial network while attacking, we still
                # need to intialize it, if the model was trained with one. Otherwise, some
                # keys of the state dict could not be mapped
                adv_config = config["adv"] if "adv" in config else dict()
                adv_model_config = None

                if use_adv_network:
                    ld = adv_config.get("latent_dropout")
                    adv_model_config = {"grad_scaling": adv_config["grad_scaling"],
                                        "latent_dropout": ld if ld else 0.5,
                                        "adversaries": [adv_config["dims"]] * adv_config["n_adv"]}

                # Initialize model
                pretrained_model = MultiVAEAdv(**config["model"],
                                               use_adv_network=use_adv_network,
                                               adv_config=adv_model_config)

                # Load pretrained model that we want to evaluate
                pretrained_model.load_state_dict(torch.load(model_path))
                pretrained_model.use_adv_network = False

                # deactivate adversaries as they don't change the metrics
                pretrained_model.to(device)
                pretrained_model.eval()

                log_str = os.path.abspath(os.path.join(run_dir, os.pardir, os.pardir))
                run_name = os.path.split(os.path.abspath(run_dir))[1]
                attack_single(log_dir=log_str, run_name=run_name, pretrained_model=pretrained_model,
                              attacker_config=attacker_config["model"], attacker_opt=attacker_config["opt"],
                              tr_loader=tr_loader, vd_loader=vd_loader, te_loader=te_loader,
                              device=device, log_batch_results_every=TR_LOG_IN_BETWEEN_EPOCH_EVERY,
                              verbose=verbose)


        print("\n")
        print("=" * 60)
        print(f"Starting attacking for {len(run_dict)} different runs")
        print("=" * 60, "\n")
        # Running hyperparameter search
        Parallel(n_jobs=n_devices, verbose=11)(
            delayed(process)(run_dir, run_data, verbose=len(run_dict) == 1)
            for run_dir, run_data in fold_dict[fold_n].items())
