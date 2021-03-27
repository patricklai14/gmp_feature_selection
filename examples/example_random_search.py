from ase import Atoms
from ase.calculators.emt import EMT
from ase.io.trajectory import Trajectory
from ase.io import read

import numpy as np
import pandas as pd

import argparse
import copy
import os
import pdb
import pickle

from model_eval import model_evaluation
from gmp_feature_selection import random_search

def main():
    dir_prefix = "/storage/home/hpaceice1/plai30/sandbox"
    parallel_workspace = os.path.join(dir_prefix, "pace/parallel_workspace")
    OUTPUT_DIR = os.path.join(dir_prefix, "output")

    #setup dataset
    np.random.seed(3)
    distances = np.linspace(2, 5, 500)
    images = []
    for i in range(len(distances)):
        l = distances[i]
        image = Atoms(
            "CuCO",
            [
                (-l * np.sin(0.65), l * np.cos(0.65), np.random.uniform(low=-4.0, high=4.0)),
                (0, 0, 0),
                (l * np.sin(0.65), l * np.cos(0.65), np.random.uniform(low=-4.0, high=4.0))
            ],
        )

        image.set_cell([10, 10, 10])
        image.wrap(pbc=True)
        image.set_calculator(EMT())
        images.append(image)

    elements = ["Cu","C","O"]
    atom_gaussians = {"C": os.path.join(dir_prefix, "config/MCSH_potential/C_coredensity_5.g"),
                      "O": os.path.join(dir_prefix, "config/MCSH_potential/O_totaldensity_7.g"),
                      "Cu": os.path.join(dir_prefix, "config/MCSH_potential/Cu_totaldensity_5.g")}
    data = model_evaluation.dataset(elements, images, atom_gaussians=atom_gaussians)

    #set up evaluation parameters
    model_eval_params = model_evaluation.get_model_eval_params(
                            fp_type="gmp", eval_type="k_fold_cv", eval_num_folds=3, eval_cv_iters=1, 
                            nn_layers=3, nn_nodes=20, nn_learning_rate=1e-3, nn_batch_size=32, nn_epochs=1000)

    selector = random_search.random_search(data, model_eval_params)
    selector.run(num_trials=3, enable_parallel=True, parallel_workspace=parallel_workspace, time_limit="00:40:00", 
                 mem_limit=2, conda_env="amptorch", seed=1)

    print("Lowest error: {}".format(selector.best_error))
    print("Best params: {}".format(selector.best_params))

    


if __name__ == "__main__":
    main()

