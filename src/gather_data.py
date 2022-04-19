import os

import torch
import numpy as np
from tqdm import trange

from conf import EXP_SEED
from src.modules.multi_vae import MultiVAEAdv
from src.utils.gather import gather_model_data, gather_dataset_stats, show_interaction_stats
from src.utils.helper import reproducible, load_config_eval, pickle_dump
from src.utils.nn_unils import get_datasets_and_loaders, adjust_result_dir, checkout_run_dict, run_to_fold_dict
from src.utils.input_validation import parse_input

TOP_K_RECOMMENDATIONS = 100

if __name__ == '__main__':
    input_config = parse_input("gathering data", options=["run", "experiment", "nfolds", "gpus", "ncores",
                                                          "split", "use_tensorboard"])

    run_dict = input_config.run_dict

    if not checkout_run_dict(run_dict):
        exit()
    fold_dict = run_to_fold_dict(run_dict)

    for fold_n in trange(input_config["nfolds"], desc='folds'):

        # Setting seed for reproducibility
        reproducible(EXP_SEED)

        # Load data
        dataset_and_loaders = get_datasets_and_loaders(dataset_name=input_config["dataset"], fold=fold_n,
                                                       splits=("train", "val", "test"), n_workers=input_config.ncores,
                                                       run_parallel=False, oversample_train=False)
        data_set, data_loader = dataset_and_loaders[input_config.split]
        print(f"{input_config.split} dataset containts {len(data_set)} samples")
        print("Data loaded\n")

        print("Gathering statistics about the dataset")
        user_count, item_interaction_count, n_male_interactions, n_female_interactions = \
            gather_dataset_stats(dataset_and_loaders)

        show_interaction_stats(user_count,
                               n_male_interactions,
                               n_female_interactions)

        # For popularity metrics
        print("Calculating item ranking for current dataset")
        count_total = np.sum([v for d in item_interaction_count.values() for g, v in d.items()])
        track_ranking_indices = np.flip(np.argsort(count_total))
        print("Ranking complete")

        for run_dir, run_data in fold_dict[fold_n].items():
            config, use_adv_network = load_config_eval(run_data["config_file"])

            result_dir = adjust_result_dir(run_dir, lambda d: f"{d}_{input_config.split}_features")
            print(f"Results will be stored in '{result_dir}'")
            os.makedirs(result_dir, exist_ok=True)

            # Saving results
            print("Saving results")
            results = {
                "user_count": user_count,
                "item_ranking": track_ranking_indices,
                "item_interaction_count": item_interaction_count,
                "n_male_interactions": n_male_interactions,
                "n_female_interactions": n_female_interactions
            }
            pickle_dump(results, os.path.join(result_dir, f"dataset_stats.pkl"))

            for model_name, model_path in run_data["models"]:
                print(f"\nGathering data for '{run_dir}' and model '{model_name}'")

                # Setting seed for reproducibility
                reproducible(EXP_SEED)

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
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                model.use_adv_network = False

                # Deactivate adversaries as they don't change the metrics
                device = input_config.devices[0]  # run only on first device for now
                model.to(device)
                model.eval()

                res = gather_model_data(model, device, data_loader, TOP_K_RECOMMENDATIONS)
                z, traits, top_k_indices, n_interactions, top_k_indices_rec = res

                # Saving results
                print("Saving results")
                results = {
                    "latent": z,
                    "traits": traits,
                    "top_k_true": top_k_indices,
                    "top_k_rec": top_k_indices_rec,
                    "n_interactions": n_interactions,
                }
                pickle_dump(results, os.path.join(result_dir, f"{model_name}_results.pkl"))
