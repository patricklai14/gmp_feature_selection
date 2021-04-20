import numpy as np
import pandas as pd

import argparse
import copy
import pathlib
import pdb
import pickle

from model_eval import model_evaluation
from gmp_feature_selection import backward_elimination

def main():
    parser = argparse.ArgumentParser(description="Run feature selection")
    parser.add_argument("--seed", type=int, required=True,
                        help="random seed to use")

    args = parser.parse_args()
    args_dict = vars(args)

    np.random.seed(args_dict["seed"])

    run_name = "back_elim_groups_run_{}".format(args_dict["seed"])

    top_dir = pathlib.Path("/storage/home/hpaceice1/plai30/sandbox")

    #setup dataset
    train_data_file = top_dir / "data/aspirin/aspirin_train_data_1.p"
    test_data_file = top_dir / "data/aspirin/aspirin_test_data_1.p"
    train_imgs = pickle.load(open(train_data_file, "rb"))
    test_imgs = pickle.load(open(test_data_file, "rb"))
 
    elements = ["C","H","O"]

    #only use training images for feature selection
    train_data_start_idx = (args_dict["seed"] - 1) * 2500
    val_data_start_idx = train_data_start_idx + 2000
    val_data_end_idx = val_data_start_idx + 500

    print("train_data_start_idx: {}, val_data_start_idx: {}, val_data_end_idx: {}".format(
            train_data_start_idx, val_data_start_idx, val_data_end_idx))
    data = model_evaluation.dataset(train_images=train_imgs[train_data_start_idx:val_data_start_idx], 
                                    test_images=train_imgs[val_data_start_idx:val_data_end_idx])

    sigmas = [0.25, 1.0, 2.0]
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

    base_config = {
        "name": run_name,
        "evaluation_type": "train_test_split",
        "seed": args_dict["seed"],
        "amptorch_config": amptorch_config
    }

    curr_dir = pathlib.Path(__file__).parent.absolute()
    workspace = curr_dir / "workspace_search_{}".format(run_name)
    back_elim = backward_elimination.backward_elimination(data, base_config)
    back_elim.run(sigmas, stop_change_pct=0.3, enable_parallel=True, parallel_workspace=workspace, 
                  time_limit="03:00:00", mem_limit=2, conda_env="amptorch")

    #print results
    print("min error: {}, config:{}".format(back_elim.get_best_error(), back_elim.get_best_params()))
    print("backward elimination stats: {}".format(back_elim.get_stats()))

    #compute test error for best params chosen by backward elimination
    print("Evaluating best model")
    num_eval_trials = 5
    test_configs = []
    test_datasets = []
    for i in range(num_eval_trials):
        curr_test_config = copy.deepcopy(back_elim.get_best_params())
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

