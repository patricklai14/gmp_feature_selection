import numpy as np
import pandas as pd

import copy
import os
import pathlib
import pdb

import model_eval
import gmp_feature_selection as gfs
#from gmp_feature_selection import constants, gmp_feature_selector

class random_search(gfs.gmp_feature_selector):
    def __init__(self, data, model_eval_params):
        super().__init__(data, model_eval_params)

    def run(self, num_trials, enable_parallel=True, parallel_workspace=None, time_limit="00:30:00", mem_limit=2, 
            conda_env="amptorch", seed=1):
        np.random.seed(seed)

        base_params = model_eval.model_evaluation.get_model_eval_params(
                       name="base", fp_type="gmp", seed=seed)
        base_params = model_eval.utils.merge_params(base_params, self.model_eval_params)

        trial_params = []
        for i in range(num_trials):
            curr_trial_params = copy.deepcopy(base_params)

            #generate random MCSH parameters
            cutoff = np.random.uniform(low=1., high=20.)

            num_sigmas = np.random.randint(low=1, high=10)
            sigmas = np.logspace(np.log10(0.02), np.log10(1.0), num=num_sigmas).tolist()

            num_orders = np.random.randint(low=1, high=11)
            gmp_orders = np.random.choice(10, num_orders, replace=False)
            gmp_orders.sort()
            gmp_order_params = {str(i): gfs.groups_by_order[i] for i in gmp_orders}

            curr_trial_params[model_eval.constants.CONFIG_JOB_NAME] = str(i)
            curr_trial_params[model_eval.constants.CONFIG_CUTOFF] = cutoff
            curr_trial_params[model_eval.constants.CONFIG_SIGMAS] = sigmas
            curr_trial_params[model_eval.constants.CONFIG_MCSH_GROUPS] = gmp_order_params

            trial_params.append(curr_trial_params)


        results = model_eval.model_evaluation.evaluate_models(
                    self.data, config_dicts=trial_params, enable_parallel=enable_parallel, 
                    workspace=parallel_workspace, time_limit=time_limit, mem_limit=mem_limit, conda_env=conda_env)

         
        min_error = -1.
        best_params = None
        for i in range(len(results)):
            print("Error for trial {}: {}".format(i, results[i].test_error))

            if best_params is None or results[i].test_error < min_error:
                min_error = results[i].test_error
                best_params = trial_params[i]

        self.best_error = min_error
        self.best_params = best_params
        

