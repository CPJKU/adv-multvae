import os
import re
from collections import defaultdict

from torch.utils.data import DataLoader

from conf import BATCH_SIZE, N_WORKERS, DEMO_TRAITS, MOVIELENS_PATH, OUT_DIR, \
    DATA_PATH, DEMO_PATH, EXP_SEED, MAX_FOLDS, ACCEPTABLE_MODEL_DIRS
from src.data.FairDatasets import FairLFM2bDataset, FairMovieLensDataset
from joblib.externals.loky.backend.context import get_context

from src.data.data_preparation import ensure_make_data
from src.data.lfm_data_splitter import DataSplitter


def dict_to_writer_format(d: dict):
    """
    Tensorboard does not allow lists for the params in its "add_hparams" function. Therefore, convert them
    to string instead
    :param d:
    :return:
    """
    writer_d = {}
    for k, v in d.items():
        writer_d[k] = v if not isinstance(v, list) else "[{0:s}]".format(",".join(map(str, v)))
    return writer_d


def get_datasets_and_loaders(dataset_name, fold, splits=("train",), batch_size=BATCH_SIZE, n_workers=N_WORKERS,
                             oversample_train=False, shuffle_train=True, run_parallel=False, traits=DEMO_TRAITS,
                             transform=None, random_state=EXP_SEED):
    data_dir, demo_dir, ds = None, None, None
    if dataset_name == "lfm2b":
        ds = DataSplitter(DATA_PATH, DEMO_PATH, out_dir=OUT_DIR, cv_n_folds=MAX_FOLDS)
        demo_dir, data_dir, _, _ = ds.get_paths(fold_n=fold)
        if oversample_train:
            ds.up_sample_train_data_path(demo_dir, data_dir, 'gender')

    elif dataset_name == "movielens":
        ensure_make_data(MOVIELENS_PATH, n_folds=MAX_FOLDS, target_path=MOVIELENS_PATH, random_state=random_state)

    dataset_loader_dict = {}
    for split in splits:
        is_train_split = split == "train"

        args = {"split": split, "up_sample": is_train_split and oversample_train,
                "traits": traits, "transform": transform}

        if dataset_name == "lfm2b":
            dataset = FairLFM2bDataset(data_dir=data_dir, demo_dir=demo_dir, data_splitter=ds, **args)
        else:
            dataset = FairMovieLensDataset(data_dir=os.path.join(MOVIELENS_PATH, str(fold)), **args)

        # for multiprocessing (with joblib) we need to set a different multiprocessing context
        # https://github.com/pytorch/pytorch/issues/44687
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            shuffle=is_train_split and shuffle_train, pin_memory=True,
                            multiprocessing_context=get_context("loky") if run_parallel else None)
        dataset_loader_dict[split] = (dataset, loader)

    return dataset_loader_dict


def adjust_result_dir(run_dir, change_fn):
    for d in ACCEPTABLE_MODEL_DIRS:
        # glob pattern to regex pattern
        rd = d.replace("*", ".*?")
        regex_dir_sep = "\\" + os.path.sep
        rd = regex_dir_sep + "(" + rd + ")" + regex_dir_sep
        if res := re.search(rd, run_dir):
            dname = res[1]
            dname_replacement = change_fn(dname)

            print(f"replacing '{dname}' with '{dname_replacement}'")

            # create own directory for results to keep folders nicely separated and clean
            return run_dir.replace(os.path.sep + dname + os.path.sep,
                                   os.path.sep + dname_replacement + os.path.sep)
    return run_dir


def checkout_run_dict(run_dict):
    if len(run_dict) > 0:
        print("Runs to attack are")
        for k in run_dict.keys():
            print(k)
        print()
        return True

    print("No (valid) runs found, canceling validation!\n")
    return False


def run_to_fold_dict(run_dict):
    fold_dict = defaultdict(lambda: dict())
    # split runs based on the fold they are in
    for run_dir, run_data in run_dict.items():
        fold_nr = extract_fold_nr(run_dir)
        if fold_nr is not None:
            fold_dict[fold_nr][run_dir] = run_data
        else:
            print(f"Could not determine on which fold to evaluate '{run_dir}' on. it is therefore ignored!")
    return dict(fold_dict)


def extract_fold_nr(path):
    regex_dir_sep = "\\" + os.path.sep
    pattern = regex_dir_sep + r"(\d{1})" + regex_dir_sep
    if res := re.search(pattern, path):
        return int(res[1])
    return None
