# Unlearning Protected User Attributes in Recommendations with Adversarial Training
This repository accompanies the paper 
`Unlearning Protected User Attributes in Recommendations with Adversarial Training`
by Christian Ganhör, David Penz, Navid Rekabsaz, Oleg Lesota and Markus Schedl.

Below we give an overview of the most important aspects to reproduce our results.

## Content
- [Source code](#source-code)
- [Running experiments](#running-experiments)
    - [Preconditions](#preconditions)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [TensorBoard](#tensorboard)

## Supported datasets
Currently, this project focuses on two datasets
- [LFM2b-bias](http://www.cp.jku.at/datasets/LFM-2b/)
- [MovieLens-1m](https://grouplens.org/datasets/movielens/1m/)

which both contain user-sensitive features such as
**gender**, **age** and **country** information.

**Note:** Currently, only the **gender** attribute is supported in the experiments.

Both datasets need to be preprocessed, before they can be used in this project.  
For preprocessing the **LFM2b** dataset, please check out the preprocessing as it was done in 
the datasets [corresponding paper](https://github.com/CPJKU/recommendation_systems_fairness#setup-the-data).

For the **MovieLens-1m** dataset, check out the notebook 
[/notebooks/preprocess_movielens_1m](notebooks/preprocess_movielens_1m.ipynb). 

After the datasets are prepared, adjust the specific paths in the global config file [conf.py](conf.py)
accordingly.

### Additional datasets
Additional datasets can be used by 
1) applying similar preprocessing (preferably as in [/notebooks/preprocess_movielens_1m](notebooks/preprocess_movielens_1m.ipynb))
2) adding additional path variables in ```conf.py```
3) adjusting the supported datasets in ```src/utils/input_validation.py```
4) extend the 'if'-cases by the new dataset in ```src/utils/nn_utils/get_datasets_and_loaders()```

## Running experiments
#### Preconditions
- setup environment
    - ```conda env create -f bias_research.yml```  
      (```bias_research_win.yml``` for Windows)

    - ```conda activate bias-research```

    - ```python3 setup.py develop```

#### Running experiments
Please check out the dedicated page on [EXPERIMENTS](EXPERIMENTS.md)  
(You can also find the descriptions of the used training configurations there.)

#### Tensorboard
Many important aspects of training and validation are logged via TensorBoard. They can 
be viewed by opening TensorBoard in the results folder for a certain experiment.

These results folders are per default created in [results/](results),
in which each dataset has a separate folder. To change this path, please adjust the variable 
`LOG_DIR` in `conf.py`.

1) move into the experiments results folder, e.g.  
   ```cd results/lfm2b/vae/standard--2021-07-15 14:14:25.366509```

2) open TensorBoard  
   ```tensorboard --logdir=./```
