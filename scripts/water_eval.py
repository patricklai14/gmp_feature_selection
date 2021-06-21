import numpy as np
import pandas as pd

import argparse
import copy
import json
import pathlib
import pdb
import pickle

from model_eval import constants, model_evaluation
from gmp_feature_selection import random_search

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on water dataset")
    parser.add_argument("--config", type=str, required=True,
                        help="config file containing gmp groups")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    args = parser.parse_args()
    args_dict = vars(args)

    top_dir = pathlib.Path("/storage/home/hcoda1/7/plai30")
    curr_dir = pathlib.Path(__file__).parent.absolute()

    config_path = args_dict["config"]
    
    MCSHs = json.load(open(config_path, "r"))

    seed = 1
    config_file = args_dict["config"].split('/')[-1]
    ext_start = config_file.find(".json")
    run_name = "eval_only_{}_lr_{}_batch_{}".format(config_file[:ext_start], args_dict["lr"], args_dict["batch_size"])

    elements = ["H","O"]
    gaussians_dir = top_dir / "sandbox/config/valence_gaussians"
    gmp_params = {
        "atom_gaussians": {
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
            "num_layers": 5, 
            "num_nodes": 50, 
            "batchnorm": False
        },
        "optim": {
            "gpus": 0,
            "force_coefficient": 0.0,
            "lr": args_dict["lr"],
            "batch_size": args_dict["batch_size"],
            "epochs": 1000,
            "loss": "mae"
            #"scheduler": {"policy": "StepLR", "params": {"step_size": 500, "gamma": 0.2}}
        },
        "dataset": {
            "val_split": 0.2,
            "elements": elements,
            "fp_scheme": "mcsh",
            "fp_params": gmp_params,
            "save_fps": False,
            "scaling": {
                #"type": "standardize"
                "type": "normalize", 
                "range": (0, 1),
                "elementwise": False
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
        "amptorch_config": amptorch_config,
        constants.CONFIG_EVAL_LOSS_TYPE: constants.CONFIG_LOSS_TYPE_ATOM_MAE
    }

    #compute test error for best params chosen by random search
    print("Evaluating model")
    test_configs = []
    curr_test_config = copy.deepcopy(base_config)
    curr_test_config["name"] = run_name
    test_configs.append(curr_test_config)

    curr_train_data_file = str(top_dir / "p-amedford6-0/data/water/train.traj")
    curr_test_data_file = str(top_dir / "p-amedford6-0/data/water/test.traj")
    curr_dataset = model_evaluation.dataset(train_data_files=[curr_train_data_file], test_data_files=[curr_test_data_file])

    workspace = curr_dir / "workspace_{}".format(run_name)
    results = model_evaluation.evaluate_models(dataset=curr_dataset, config_dicts=test_configs,
                                               enable_parallel=True, workspace=workspace,
                                               time_limit="20:00:00", mem_limit=2, conda_env="amptorch")

    #print results
    errors = [metrics.test_error for metrics in results]
    print("Test errors: {}".format(errors))
    print("MAE: {}".format(np.mean(errors)))



if __name__ == "__main__":
    main()

