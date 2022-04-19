import os

import torch
from tqdm import trange

from conf import EXP_SEED
from src.modules.multi_vae import MultiVAEAdv
from src.modules.polylinear import PolyLinear
from src.utils.input_validation import parse_input
from src.utils.gather import gather_model_atk_data
from src.utils.helper import reproducible, load_config_eval, pickle_dump, json_load
from src.utils.nn_unils import get_datasets_and_loaders, adjust_result_dir, checkout_run_dict, run_to_fold_dict


if __name__ == '__main__':
    input_config = parse_input("gathering data", options=["run", "experiment", "nfolds", "gpus", "ncores",
                                                          "split", "use_tensorboard"], access_as_properties=False)

    run_dict = input_config["run_dict"]

    if not checkout_run_dict(run_dict):
        exit()
    fold_dict = run_to_fold_dict(run_dict)

    for fold_n in trange(input_config["nfolds"], desc='folds'):

        # Setting seed for reproducibility
        reproducible(EXP_SEED)

        # Load data
        dataset_and_loaders = get_datasets_and_loaders(dataset_name=input_config["dataset"], fold=fold_n,
                                                       splits=("train", "val", "test"), n_workers=input_config["ncores"],
                                                       run_parallel=False, oversample_train=False)
        data_set, data_loader = dataset_and_loaders[input_config["split"]]
        print(f"{input_config['split']} dataset containts {len(data_set)} samples")
        print("Data loaded\n")

        for run_dir, run_data in fold_dict[fold_n].items():
            config, use_adv_network = load_config_eval(run_data["config_file"])

            result_dir = adjust_result_dir(run_dir, lambda d: f"{d}_{input_config['split']}_features")
            print(f"Results will be stored in '{result_dir}'")
            os.makedirs(result_dir, exist_ok=True)

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
                print("model path", model_path)
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                model.use_adv_network = False

                # Deactivate adversaries as they don't change the metrics
                device = input_config["devices"][0]  # run only on first device for now
                model.to(device)
                model.eval()

                # Retrieve attacker directory
                attacker_suffix_map = {
                    "best_model_adv": "adv",
                    "adv_model_epoch_last": "last",
                    "best_model_ndcg": "ndcg",
                    "best_model_recall": "recall"
                }
                attacker_dir = adjust_result_dir(run_dir, lambda d: f"atk").rstrip(os.path.sep)
                attacker_dir += "_" + attacker_suffix_map[model_name] + os.path.sep
                attacker_config = json_load(os.path.join(attacker_dir, "atk_config.json"))

                activation_fn = torch.nn.ReLU() if attacker_config["activation"] == "relu" else torch.nn.Tanh()
                layer_config = [model.latent] + attacker_config["size"]
                attacker_model = PolyLinear(layer_config=layer_config, activation_fn=activation_fn)
                attacker_model.load_state_dict(torch.load(os.path.join(attacker_dir, "atk_model.pt"),
                                                          map_location="cpu"))
                attacker_model.to(device)
                attacker_model.eval()

                res = gather_model_atk_data(model, attacker_model, device, data_loader)
                z, az, traits = res

                # Saving results
                print("Saving results")
                results = {
                    "latent": z,
                    "atk_output": az,
                    "traits": traits
                }
                pickle_dump(results, os.path.join(result_dir, f"{model_name}_latent_adv_results.pkl"))
