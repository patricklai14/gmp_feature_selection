import numpy as np
import pandas as pd

import argparse
import copy
import pathlib
import pdb
import pickle

from model_eval import constants, model_evaluation, utils

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on aspirin dataset")
    parser.add_argument("--max_order", type=int, required=True,
                        help="number of GMP orders to include")

    args = parser.parse_args()
    args_dict = vars(args)

    sigmas = [0.25, 1.0, 2.0]
    #sigmas = [0.25, 0.75, 1.5, 2.0]
    all_mcsh_groups = {"0": {"groups": [1]},
                       "1": {"groups": [1]},
                       "2": {"groups": [1,2]},
                       "3": {"groups": [1,2,3]},
                       "4": {"groups": [1,2,3,4]},
                       "5": {"groups": [1,2,3,4,5]},
                       "6": {"groups": [1,2,3,4,5,6,7]},
                       "7": {"groups": [1,2,3,4,5,6,7,8]},
                       "8": {"groups": [1,2,3,4,5,6,7,8,9,10]}}

    MCSHs = {}
    for order in range(args_dict["max_order"] + 1):
        order_str = str(order)
        MCSHs[order_str] = all_mcsh_groups[order_str]
        MCSHs[order_str]["sigmas"] = sigmas

    top_dir = pathlib.Path("/storage/home/hcoda1/7/plai30")
    gaussians_dir = top_dir / "sandbox/config/valence_gaussians"
    gmp_params = {
        "atom_gaussians": {
            "H": str(gaussians_dir / "H_pseudodensity_2.g"),
            "O": str(gaussians_dir / "O_pseudodensity_4.g")
        },
        "MCSHs": MCSHs,
        "cutoff": 10
    }

    elements = ["H","O"]
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
            #"lr": 1e-3,
            "batch_size": 128,
            "epochs": 500,
            "loss": "mae"
            #"scheduler": {"policy": "StepLR", "params": {"step_size": 500, "gamma": 0.5}}
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
        "evaluation_type": "train_test_split",
        "seed": 1,
        "amptorch_config": amptorch_config,
        constants.CONFIG_EVAL_LOSS_TYPE: constants.CONFIG_LOSS_TYPE_ATOM_MAE
    }
 
    run_name = "convergence_order_{}".format(args_dict["max_order"])

    curr_test_config = copy.deepcopy(base_config)
    curr_test_config["name"] = run_name
    curr_test_config["seed"] = 1
    curr_test_config["amptorch_config"]["cmd"]["seed"] = 1

    curr_train_data_file = str(top_dir / "p-amedford6-0/data/water/train.traj")
    curr_test_data_file = str(top_dir / "p-amedford6-0/data/water/test.traj")
    curr_dataset = model_evaluation.dataset(train_data_files=[curr_train_data_file], test_data_files=[curr_test_data_file])

    curr_dir = pathlib.Path(__file__).parent.absolute()
    save_model_dir = curr_dir / "{}_model_checkpoints".format(run_name)
    num_training_iters = 10
    lr = 1e-3

    for i in range(num_training_iters):
        print("Evaluating models for training iteration {}".format(i + 1))
        workspace = curr_dir / "workspace_{}_{}".format(run_name, i + 1)

        checkpoint_dirs = []
        if i != 0:
            #find checkpoint directories
            checkpoint_dirs.append(utils.get_checkpoint_dir(save_model_dir / curr_test_config["name"]))

        #if this is the last iteration, no need to save models
        if i >= num_training_iters - 1:
            save_model_dir = ""

        curr_test_config["amptorch_config"]["optim"]["lr"] = lr
        results = model_evaluation.evaluate_models(dataset=curr_dataset, config_dicts=[curr_test_config],
                                                   enable_parallel=True, workspace=workspace,
                                                   time_limit="03:00:00", mem_limit=2, conda_env="amptorch",
                                                   save_model_dir=save_model_dir, checkpoint_dirs=checkpoint_dirs)

        #print results
        errors = [metrics.test_error for metrics in results]
        print("Test errors after training iteration {}: {}".format(i + 1, errors))
        print("MAE: {}".format(np.mean(errors)))

        lr = lr * 0.5


if __name__ == "__main__":
    main()
