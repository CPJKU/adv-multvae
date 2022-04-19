import csv
import os

import numpy as np
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange

import torch
from torch.utils.tensorboard import SummaryWriter

from conf import EXP_SEED, DEMO_TRAITS
from src.utils.eval import eval_metrics
from src.modules.multi_vae import MultiVAEAdv
from src.utils.helper import reproducible, json_dump, pickle_dump, load_config_eval, flatten_config
from src.utils.nn_unils import dict_to_writer_format, get_datasets_and_loaders, adjust_result_dir, \
    run_to_fold_dict, checkout_run_dict
from src.utils.input_validation import parse_input
from src.utils.vae_training_utils import eval_adversaries

if __name__ == '__main__':
    input_config = parse_input("gathering data", options=["run", "experiment", "nfolds", "gpus", "ncores",
                                                          "split", "use_tensorboard"])

    run_dict = input_config.run_dict
    if not checkout_run_dict(run_dict):
        exit()
    fold_dict = run_to_fold_dict(run_dict)

    for fold_n in trange(input_config.nfolds, desc='folds'):

        # Setting seed for reproducibility
        reproducible(EXP_SEED)

        # Load data
        dataset_and_loaders = get_datasets_and_loaders(dataset_name=input_config.dataset, fold=fold_n,
                                                       splits=(input_config.split,), n_workers=input_config.ncores,
                                                       run_parallel=False)
        data_set, data_loader = dataset_and_loaders[input_config.split]
        print(f"{input_config.split} dataset containts {len(data_set)} samples")
        print("Data Loaded\n")

        for run_dir, run_data in fold_dict[fold_n].items():
            config, use_adv_network = load_config_eval(run_data["config_file"])

            for model_name, model_path in run_data["models"]:
                print(f"\nRunning validation for \n{run_dir} \nand model \n{model_name}")
                # create own directory for results to keep folders nicely separated and clean
                result_dir = adjust_result_dir(run_dir, lambda d: f"{d}_{input_config.split}_eval")
                print(f"Results will be stored in '{result_dir}'")
                os.makedirs(result_dir, exist_ok=True)

                adv_config = config["adv"] if "adv" in config else dict()
                adv_model_config = None

                if use_adv_network:
                    ld = adv_config.get("latent_dropout")
                    adv_model_config = {"grad_scaling": adv_config["grad_scaling"],
                                        "latent_dropout": ld if ld else 0.5,
                                        "adversaries": [adv_config["dims"]] * adv_config["n_adv"]}

                # Initialize model
                model = MultiVAEAdv(**config["model"],
                                    use_adv_network=use_adv_network,
                                    adv_config=adv_model_config)

                # Load pretrained model that we want to evaluate
                model.load_state_dict(torch.load(model_path))

                device = input_config.devices[0]  # run only on first device for now
                model.to(device)
                model.eval()

                adv_loss_fn = CrossEntropyLoss()

                all_y = []
                all_logits = []
                all_adv_bal_accs = []
                with torch.no_grad():
                    for x, y, adv_targets in tqdm(data_loader, desc="Evaluating..."):
                        x = x.to(device)
                        adv_targets = adv_targets.to(device, dtype=torch.long)  # BCE loss requires data of type long

                        logits, _, adv_logits = model(x)
                        _, adv_bal_acc = eval_adversaries(adv_logits, adv_targets, adv_loss_fn)

                        # Removing items from training data
                        logits[x.nonzero(as_tuple=True)] = .0

                        # Fetching all predictions and ground_truth labels
                        all_y.append(y.cpu().numpy())
                        all_logits.append(logits.cpu().numpy())
                        all_adv_bal_accs.append(adv_bal_acc.item())

                true = np.concatenate(all_y)
                preds = np.concatenate(all_logits)
                bal_acc = np.mean(all_adv_bal_accs)

                full_metrics = dict()
                full_raw_metrics = dict()

                print("Calculating metrics")
                for trait in DEMO_TRAITS:
                    user_groups = data_set.user_groups_per_trait[trait]

                    _, metrics, metrics_raw = eval_metrics(preds=preds,
                                                           true=true,
                                                           tag=input_config.split,
                                                           user_groups=user_groups
                                                           )

                    full_metrics.update(metrics)
                    full_raw_metrics.update(metrics_raw)

                # Saving results and predictions
                print("Saving results")
                json_dump(full_metrics, os.path.join(result_dir, f"{model_name}_full_metrics.json"))
                pickle_dump(full_metrics, os.path.join(result_dir, f"{model_name}_full_metrics.pkl"))
                pickle_dump(full_raw_metrics, os.path.join(result_dir, f"{model_name}_full_raw_metrics.pkl"))

                with open(os.path.join(result_dir, f"{model_name}_adv_scores.csv"), "w", newline="") as fh:
                    csv_writer = csv.writer(fh, delimiter=";", )
                    csv_writer.writerow(["adversarial_balanced_accuracy"])
                    csv_writer.writerow([bal_acc])

                if input_config.use_tensorboard:
                    # Logging hyperparams and metrics
                    writer = SummaryWriter(result_dir)

                    fconfig = flatten_config(config)
                    fconfig["model_name"] = model_name

                    writer.add_hparams({**dict_to_writer_format(fconfig), 'fold_n': fold_n}, full_metrics)
                    writer.flush()
