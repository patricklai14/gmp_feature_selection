import numpy as np
import pandas as pd

import copy
import os
import pathlib
import pdb

import model_eval
from gmp_feature_selection import constants, gmp_feature_selector

class backward_elimination(gmp_feature_selector.gmp_feature_selector):
    def __init__(self, data, model_eval_params):
        super().__init__(data, model_eval_params)

    def run(output_dir, data, num_trials, enable_parallel, parallel_workspace=None, seed=1):
        np.random.seed(seed)

        base_params = model_eval.model_evaluation.get_model_eval_params(
                       "base", "gmp", "k_fold_cv", eval_cv_iters=2, eval_num_folds=2, nn_layers=3,
                       nn_nodes=20, nn_learning_rate=1e-3, nn_batch_size=32, nn_epochs=1000, seed=seed)

        trial_params = []
        for i in range(num_trials):
            curr_trial_params = copy.deepcopy(base_params)

            #generate random MCSH parameters
            cutoff = np.random.uniform(low=1., high=20.)

            num_sigmas = np.random.randint(low=1, high=10)
            sigmas = np.logspace(np.log10(0.02), np.log10(1.0), num=num_sigmas).tolist()

            num_orders = np.random.randint(low=1, high=11)
            gmp_orders = np.random.choice(10, num_orders, replace=False).sort()
            gmp_order_params = {str(i): constants.groups_by_order[i] for i in gmp_orders}

            curr_trial_params[model_eval.constants.CONFIG_CUTOFF] = cutoff
            curr_trial_params[model_eval.constants.CONFIG_SIGMAS] = sigmas
            curr_trial_params[model_eval.constants.CONFIG_MCSH_GROUPS] = gmp_order_params

            trial_params.append(curr_trial_params)


        results = model_eval.model_evaluation.evaluate_models(
                    data, config_dicts=candidate_params, enable_parallel=enable_parallel, workspace=parallel_workspace,
                    time_limit="00:40:00", mem_limit=2, conda_env="amptorch")


