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
from gmp_feature_selection import backward_elimination

def main():
    args = parser.parse_args()
    args_dict = vars(args)

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
    cutoff = 8
    sigmas = (np.logspace(np.log10(0.05), np.log10(1.0), num=5)).tolist()
    model_eval_params = model_evaluation.get_model_eval_params(
                            fp_type="gmp", eval_type="k_fold_cv", eval_num_folds=2, eval_cv_iters=1, 
                            cutoff=cutoff, sigmas=sigmas, nn_layers=3, nn_nodes=20, nn_learning_rate=1e-3, 
                            nn_batch_size=32, nn_epochs=1000)

    back_elim = backward_elimination(OUTPUT_DIR, data, model_eval_params)
    back_elim.run(enable_parallel=True, parallel_workspace=parallel_workspace, seed=1)


if __name__ == "__main__":
    main()

