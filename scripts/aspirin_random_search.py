import numpy as np
import pandas as pd

import argparse
import copy
import pathlib
import pdb
import pickle

from model_eval import model_evaluation
from gmp_feature_selection import random_search

def main():
    parser = argparse.ArgumentParser(description="Run feature selection")
    parser.add_argument("--groups_pct", type=float, required=True,
                        help="size of subset of GMP groups to generate (as pct of maximum)")
    parser.add_argument("--seed", type=int, required=True,
                        help="random seed to use")

    args = parser.parse_args()
    args_dict = vars(args)

    run_name = "random_groups_{}_run_{}".format(int(args_dict["groups_pct"] * 100), args_dict["seed"])

    top_dir = pathlib.Path("/storage/home/hpaceice1/plai30/sandbox")

    #setup dataset
    train_data_file = top_dir / "data/aspirin/aspirin_train_data_1.p"
    test_data_file = top_dir / "data/aspirin/aspirin_test_data_1.p"
    train_imgs = pickle.load(open(train_data_file, "rb"))
    test_imgs = pickle.load(open(test_data_file, "rb"))
 
    elements = ["C","H","O"]

    #only use training images for feature selection
    data = model_evaluation.dataset(train_images=train_imgs[:8000], test_images=train_imgs[8000:10000])

    sigmas = [0.25, 1.0, 2.0]
    all_mcsh_groups = {"0": {"groups": [1]},
                       "1": {"groups": [1]},
                       "2": {"groups": [1,2]},
                       "3": {"groups": [1,2,3]},
                       "4": {"groups": [1,2,3,4]},
                       "5": {"groups": [1,2,3,4,5]},
                       "6": {"groups": [1,2,3,4,5,6,7]}}

    gaussians_dir = top_dir / "config/valence_gaussians"
    gmp_params = {
        "atom_gaussians": {
            "C": str(gaussians_dir / "C_pseudodensity_4.g"),
            "H": str(gaussians_dir / "H_pseudodensity_2.g"),
            "O": str(gaussians_dir / "O_pseudodensity_4.g")
        },
        "cutoff": 10
    }

    amptorch_config = {
        "model": {
            "name":"singlenn",
            "get_forces": False, 
            "num_layers": 3, 
            "num_nodes": 50, 
            "batchnorm": True
        },
        "optim": {
            "gpus": 0,
            "force_coefficient": 0.0,
            "lr": 1e-3,
            "batch_size": 256,
            "epochs": 5000,
            "loss": "mae",
        },
        "dataset": {
            "val_split": 0.2,
            "elements": elements,
            "fp_scheme": "mcsh",
            "fp_params": gmp_params,
            "save_fps": False,
            "scaling": {
                "type": "normalize", 
                "range": (0, 1),
                "elementwise":False
            }
        },
        "cmd": {
            "debug": False,
            "identifier": "test",
            "verbose": True,
            "logger": False
        }
    }

    group_pairs = []
    for order, groups_dict in all_mcsh_groups.items():
        for group in groups_dict["groups"]:
            group_pairs.append((order, group))

    target_groups_pct = args_dict["groups_pct"]
    target_num_groups = int(target_groups_pct * len(group_pairs))
    num_random_trials = 20 

    seed = args_dict["seed"]
    np.random.seed(seed)
    random_configs = []
    for i in range(num_random_trials):
        curr_indices = np.random.choice(len(group_pairs), target_num_groups, replace=False)
        curr_groups = [group_pairs[idx] for idx in curr_indices]
        curr_groups.sort()

        curr_mcsh_groups = {}
        for order, group in curr_groups:
            if order not in curr_mcsh_groups:
                curr_mcsh_groups[order] = {"groups": []}
            
            curr_mcsh_groups[order]["groups"].append(group)

        for order, order_params in curr_mcsh_groups.items():
            order_params["sigmas"] = sigmas

        curr_amptorch_config = copy.deepcopy(amptorch_config)
        curr_amptorch_config["dataset"]["fp_params"]["MCSHs"] = curr_mcsh_groups
        curr_amptorch_config["cmd"]["seed"] = seed

        curr_config = {
            "name": "{}_search_{}".format(run_name, i),
            "evaluation_type": "train_test_split",
            "seed": seed,
            "amptorch_config": curr_amptorch_config
        }

        random_configs.append(curr_config)

    curr_dir = pathlib.Path(__file__).parent.absolute()
    workspace = curr_dir / "workspace_search_{}".format(run_name)
    results = model_evaluation.evaluate_models(data, config_dicts=random_configs,
                                               enable_parallel=True, workspace=workspace,
                                               time_limit="05:00:00", mem_limit=2, conda_env="amptorch")

    #print results
    errors = [metrics.test_error for metrics in results]
    print("Test errors: {}".format(errors))

    best_idx = np.argmin(errors)
    print("min error: {}, config:{}".format(errors[best_idx], random_configs[best_idx]))


    #compute test error for best params chosen by random search
    print("Evaluating best model")
    num_eval_trials = 5
    test_configs = []
    test_datasets = []
    for i in range(num_eval_trials):
        curr_test_config = copy.deepcopy(random_configs[best_idx])
        curr_test_config["name"] = "eval_test_{}".format(i + 1)
        curr_test_config["seed"] = 1
        curr_test_config["amptorch_config"]["cmd"]["seed"] = 1

        test_configs.append(curr_test_config)

        curr_train_data_file = str(top_dir / "data/aspirin/aspirin_train_data_{}.p".format(i + 1))
        curr_test_data_file = str(top_dir / "data/aspirin/aspirin_test_data_{}.p".format(i + 1))
        curr_dataset = model_evaluation.dataset(train_data_files=[curr_train_data_file], test_data_files=[curr_test_data_file])
        test_datasets.append(curr_dataset)

    curr_dir = pathlib.Path(__file__).parent.absolute()
    workspace = curr_dir / "workspace_eval_{}".format(run_name)
    results = model_evaluation.evaluate_models(datasets=test_datasets, config_dicts=test_configs,
                                               enable_parallel=True, workspace=workspace,
                                               time_limit="05:00:00", mem_limit=2, conda_env="amptorch")

    #print results
    errors = [metrics.test_error for metrics in results]
    print("Test errors: {}".format(errors))
    print("MAE: {}".format(np.mean(errors)))



if __name__ == "__main__":
    main()

