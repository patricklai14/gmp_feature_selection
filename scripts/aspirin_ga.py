import numpy as np
import pandas as pd

import argparse
import copy
import pathlib
import pdb
import pickle

from model_eval import model_evaluation
import gmp_feature_selection as gfs

def main():
    parser = argparse.ArgumentParser(description="Run genetic algorithm on MD17 dataset")
    parser.add_argument("--seed", type=int, required=True,
                        help="random seed to use (in [1, 5])")

    args = parser.parse_args()
    args_dict = vars(args)

    np.random.seed(args_dict["seed"])
    run_name = "ga_run_{}".format(args_dict["seed"])
    top_dir = pathlib.Path("/storage/home/hcoda1/7/plai30")

    #setup dataset (use one of the training sets as validation during feature selection)
    train_data_file = top_dir / "p-amedford6-0/data/aspirin/aspirin_train_data_{}.p".format(args_dict["seed"])
    val_data_file_num = 1 if args_dict["seed"] == 5 else args_dict["seed"] + 1 
    val_data_file = top_dir / "p-amedford6-0/data/aspirin/aspirin_train_data_{}.p".format(val_data_file_num)
    #train_imgs = pickle.load(open(train_data_file, "rb"))
    val_imgs = pickle.load(open(val_data_file, "rb"))
 
    data = model_evaluation.dataset(train_data_files=[train_data_file], test_images=val_imgs[:10000])
    elements = ["C","H","O"]

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
            "epochs": 3500,
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
    workspace = curr_dir / "workspace_{}".format(run_name)

    ga = gfs.genetic_algorithm(data, base_config)
    ga.run(population_size=4, num_generations=2, target_groups_pct=0.8, max_order=7, crossover_prob=0.75, 
           mutation_prob=0.2, enable_parallel=True, parallel_workspace=workspace, time_limit="03:00:00", 
           mem_limit=2, conda_env="amptorch")

    #print results
    print("min error: {}, config:{}".format(ga.get_best_error(), ga.get_best_params()))

    #compute test error for best params chosen by the genetic algorithm
    print("Evaluating best model")
    num_eval_trials = 5
    test_configs = []
    test_datasets = []
    for i in range(num_eval_trials):
        curr_test_config = copy.deepcopy(ga.get_best_params())
        curr_test_config["name"] = "eval_{}_{}".format(run_name, i + 1)
        curr_test_config["seed"] = 1
        curr_test_config["amptorch_config"]["cmd"]["seed"] = 1

        test_configs.append(curr_test_config)

        curr_train_data_file = str(top_dir / "data/aspirin/aspirin_train_data_{}.p".format(i + 1))
        curr_test_data_file = str(top_dir / "data/aspirin/aspirin_test_data_{}.p".format(i + 1))
        curr_dataset = model_evaluation.dataset(train_data_files=[curr_train_data_file], test_data_files=[curr_test_data_file])
        test_datasets.append(curr_dataset)

    results = model_evaluation.evaluate_models(datasets=test_datasets, config_dicts=test_configs,
                                               enable_parallel=True, workspace=workspace,
                                               time_limit="05:00:00", mem_limit=2, conda_env="amptorch")

    #print results
    errors = [metrics.test_error for metrics in results]
    print("Test errors: {}".format(errors))
    print("MAE: {}".format(np.mean(errors)))



if __name__ == "__main__":
    main()

