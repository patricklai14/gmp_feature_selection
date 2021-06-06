import numpy as np
import pandas as pd

import argparse
import copy
import pathlib
import pdb
import pickle

from ase.io.trajectory import Trajectory

from model_eval import constants, model_evaluation
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
    data_dir = top_dir / "p-amedford6-0/data"

    #setup dataset (use part of the training set as validation during feature selection)
    train_data_file = str(top_dir / "p-amedford6-0/data/water/single/train.traj")
    traj = Trajectory(train_data_file)
    all_imgs = [img for img in traj]
    val_data_start = int(0.75 * len(all_imgs))
    train_imgs = all_imgs[:val_data_start]
    val_imgs = all_imgs[val_data_start:]

    data = model_evaluation.dataset(train_images=train_imgs, test_images=val_imgs)
    elements = ["H","O"]

    gaussians_dir = top_dir / "sandbox/config/valence_gaussians"
    gmp_params = {
        "atom_gaussians": {
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
            "batch_size": 128,
            "epochs": 2500,
            "loss": "mae",
            "scheduler": {"policy": "StepLR", "params": {"step_size": 500, "gamma": 0.5}}
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
        "amptorch_config": amptorch_config,
        constants.CONFIG_EVAL_LOSS_TYPE: constants.CONFIG_LOSS_TYPE_ATOM_MAE
    }

    curr_dir = pathlib.Path(__file__).parent.absolute()
    workspace = curr_dir / "workspace_{}".format(run_name)

    ga = gfs.genetic_algorithm(data, base_config)
    ga.run(population_size=20, num_generations=5, target_groups_pct=1.0, max_order=5, crossover_prob=0.75, 
           mutation_prob=0.2, enable_parallel=True, parallel_workspace=workspace, time_limit="00:30:00", 
           mem_limit=2, conda_env="amptorch", seed=args_dict["seed"])

    #print results
    print("min error: {}, config:{}".format(ga.get_best_error(), ga.get_best_params()))

    #compute test error for best params chosen by the genetic algorithm
    print("Evaluating best model")
    test_configs = []
    for i in range(1, 6):
        curr_test_config = copy.deepcopy(ga.get_best_params())
        curr_test_config["name"] = run_name + "_eval_{}".format(i)
        curr_test_config["seed"] = i
        curr_test_config["amptorch_config"]["cmd"]["seed"] = i
        test_configs.append(curr_test_config)

    test_data_file = str(top_dir / "p-amedford6-0/data/water/single/test.traj")
    curr_dataset = model_evaluation.dataset(train_data_files=[train_data_file], test_data_files=[test_data_file])

    workspace = curr_dir / "workspace_{}".format(run_name)
    results = model_evaluation.evaluate_models(dataset=curr_dataset, config_dicts=test_configs,
                                               enable_parallel=True, workspace=workspace,
                                               time_limit="01:00:00", mem_limit=2, conda_env="amptorch")

    #print results
    errors = [metrics.test_error for metrics in results]
    print("Test errors: {}".format(errors))
    print("MAE: {}".format(np.mean(errors)))




if __name__ == "__main__":
    main()

