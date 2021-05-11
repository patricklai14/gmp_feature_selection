import numpy as np
import pandas as pd

import argparse
import copy
import json
import pathlib
import pdb
import pickle

from model_eval import model_evaluation
from gmp_feature_selection import random_search

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on aspirin dataset")
    parser.add_argument("--config", type=str, required=True,
                        help="config file containing gmp groups")

    args = parser.parse_args()
    args_dict = vars(args)

    top_dir = pathlib.Path("/storage/home/hpaceice1/plai30/sandbox")
    config_path = top_dir / "scripts/groups_dicts/{}".format(args_dict["config"])
    
    MCSHs = json.load(open(config_path, "r"))

    seed = 1
    ext_start = args_dict["config"].find(".json")
    run_name = "eval_only_{}".format(args_dict["config"][7:ext_start])

    elements = ["C","H","O"]
    sigmas = [0.25, 1.0, 2.0]

    gaussians_dir = top_dir / "config/valence_gaussians"
    gmp_params = {
        "atom_gaussians": {
            "C": str(gaussians_dir / "C_pseudodensity_4.g"),
            "H": str(gaussians_dir / "H_pseudodensity_2.g"),
            "O": str(gaussians_dir / "O_pseudodensity_4.g")
        },
        "MCSHs": MCSHs,
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
            "epochs": 1000,
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
            "logger": False,
            "seed": seed
        }
    }

    base_config = {
        "evaluation_type": "train_test_split",
        "seed": seed,
        "amptorch_config": amptorch_config
    }

    #compute test error for best params chosen by random search
    print("Evaluating best model")
    num_eval_trials = 5
    test_configs = []
    test_datasets = []
    for i in range(num_eval_trials):
        curr_test_config = copy.deepcopy(base_config)
        curr_test_config["name"] = "{}_{}".format(run_name, i + 1)

        test_configs.append(curr_test_config)

        curr_train_data_file = str(top_dir / "data/aspirin/aspirin_train_data_{}.p".format(i + 1))
        curr_test_data_file = str(top_dir / "data/aspirin/aspirin_test_data_{}.p".format(i + 1))
        #curr_train_data_file = str(top_dir / "data/aspirin/full/aspirin_train_data_{}.p".format(i + 1))
        #curr_test_data_file = str(top_dir / "data/aspirin/full/aspirin_test_data_{}.p".format(i + 1))
        curr_dataset = model_evaluation.dataset(train_data_files=[curr_train_data_file], test_data_files=[curr_test_data_file])
        test_datasets.append(curr_dataset)

    curr_dir = pathlib.Path(__file__).parent.absolute()
    workspace = curr_dir / "workspace_eval_{}".format(run_name)
    results = model_evaluation.evaluate_models(datasets=test_datasets, config_dicts=test_configs,
                                               enable_parallel=True, workspace=workspace,
                                               time_limit="02:00:00", mem_limit=2, conda_env="amptorch", num_train_iters=5)

    #print results
    errors = [metrics.test_error for metrics in results]
    print("Test errors: {}".format(errors))
    print("MAE: {}".format(np.mean(errors)))



if __name__ == "__main__":
    main()

