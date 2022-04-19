import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from src.utils.eval import top_k


def gather_dataset_stats(dataset_and_loaders):
    # Determine item ranking for current datasets
    user_count = defaultdict(lambda: dict())
    item_interaction_count = defaultdict(lambda: dict())
    n_male_interactions = 0
    n_female_interactions = 0
    for s, (ds, dl) in dataset_and_loaders.items():
        n_male, n_female = 0, 0
        count_male = np.zeros(shape=(ds.n_items,))
        count_female = np.zeros(shape=(ds.n_items,))

        for inp, tar, gender in dl:
            male_user_inp = inp[(gender == 1).flatten()].numpy()
            female_user_inp = inp[(gender == 0).flatten()].numpy()

            count_male += male_user_inp.sum(axis=0)
            count_female += female_user_inp.sum(axis=0)

            # Except for the train set, input and target are not the same
            if s != "train":
                male_user_tar = tar[(gender == 1).flatten()].numpy()
                female_user_tar = tar[(gender == 0).flatten()].numpy()

                count_male += male_user_tar.sum(axis=0)
                count_female += female_user_tar.sum(axis=0)

            n_male += len(male_user_inp)
            n_female += len(female_user_inp)

        # Store results
        user_count[s]["male"] = n_male
        user_count[s]["female"] = n_female

        item_interaction_count[s]["male"] = count_male
        item_interaction_count[s]["female"] = count_female

        n_male_interactions += count_male.sum()
        n_female_interactions += count_female.sum()
    return dict(user_count), dict(item_interaction_count), n_male_interactions, n_female_interactions


def show_interaction_stats(user_count, n_male_interactions, n_female_interactions):
    print("\nTotal male interactions: ", n_male_interactions)
    print("Total female interactions: ", n_female_interactions)

    n_male_users = np.sum([uc["male"] for uc in user_count.values()])
    n_female_users = np.sum([uc["female"] for uc in user_count.values()])

    print("\nNumber of male users:", n_male_users)
    print("Number of female users:", n_female_users)

    mean_male_interactions = n_male_interactions / n_male_users
    mean_female_interactions = n_female_interactions / n_female_users

    print("\nMean male interactions: {:.4f}".format(mean_male_interactions))
    print("Mean female interactions: {:.4f}".format(mean_female_interactions))
    print("Ratio: {:.4f}\n".format(mean_male_interactions / mean_female_interactions))
    return mean_male_interactions, mean_female_interactions


def gather_model_data(model, device, data_loader, n_top_k):
    all_z = []
    all_traits = []
    all_top_k_indices = []
    all_n_interactions = []
    all_top_k_indices_rec = []
    with torch.no_grad():
        for x, y, traits in tqdm(data_loader, desc="Gathering model data..."):
            x = x.to(device)
            z, _ = model.encoder_forward(x)
            o = model.decoder(z)

            # Collect results
            all_z.append(z.cpu().numpy())
            all_traits.append(traits.cpu().numpy())

            # Gather actual top_k
            y = y.numpy()
            _, true_top_k, _ = top_k(y, k=n_top_k)
            all_top_k_indices.append(true_top_k)

            # may may have less than TOP_K interactions, so we need to save
            # the actual number for later investigation
            all_n_interactions.append(y.sum(axis=1))

            # Determine recommended tracks
            o = o.cpu().numpy()
            _, recommended_top_k, _ = top_k(o, k=n_top_k)
            all_top_k_indices_rec.append(recommended_top_k)

    all_z = np.concatenate(all_z)
    all_traits = np.concatenate(all_traits)
    all_top_k_indices = np.concatenate(all_top_k_indices)
    all_n_interactions = np.concatenate(all_n_interactions)
    all_top_k_indices_rec = np.concatenate(all_top_k_indices_rec)
    return all_z, all_traits, all_top_k_indices, all_n_interactions, all_top_k_indices_rec


def gather_model_atk_data(model, attacker, device, data_loader):
    all_z = []
    all_az = []
    all_traits = []
    with torch.no_grad():
        for x, y, traits in tqdm(data_loader, desc="Gathering model data..."):
            x = x.to(device)
            z, _ = model.encoder_forward(x)
            az = attacker(z)

            # Collect results
            all_z.append(z.cpu().numpy())
            all_az.append(az.cpu().numpy())
            all_traits.append(traits.cpu().numpy())

    all_z = np.concatenate(all_z)
    all_az = np.concatenate(all_az)
    all_traits = np.concatenate(all_traits)
    return all_z, all_az, all_traits
