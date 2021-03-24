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

    def run(self, enable_parallel=False, parallel_workspace=None, seed=None, output_dir=None):
        #setup baseline MCSH params
        base_order = 9 #number of orders to include by default
        base_group_params = {str(i): constants.groups_by_order[i] for i in range(base_order + 1)}

        base_group_params = model_eval.model_evaluation.get_model_eval_params(
                                name="base", fp_type="gmp", mcsh_groups=base_group_params, seed=seed)
        base_params = model_eval.utils.merge_params(base_group_params, self.model_eval_params)

        #get baseline performance
        print("Testing base params: {}".format(base_params))
        base_train_mse, base_test_mse = model_eval.model_evaluation.evaluate_model(base_params, self.data)
        print("Base test MSE: {}".format(base_test_mse))

        stop_change_pct = 0.15
        prev_test_mse = base_test_mse
        prev_group_params = copy.deepcopy(base_group_params)

        MSEs = [base_test_mse]
        orders_removed = [-1]

        print("Backward elimination params: stop_change_pct={}".format(
                stop_change_pct))

        #perform backward elimination
        while True:
            curr_min_test_mse = 1000000.
            curr_best_order = -1
            curr_best_group_params = None

            candidate_orders = []
            candidate_params = []
            
            print("Creating configs for processing on Pace")
            for order, order_params in prev_group_params.items():
                group_params_candidate = copy.deepcopy(prev_group_params)
                order_str = str(order)
                del group_params_candidate[order_str]

                eval_params_candidate = copy.deepcopy(base_params)
                eval_params_candidate[model_eval.constants.CONFIG_JOB_NAME] = str(order)
                eval_params_candidate[model_eval.constants.CONFIG_MCSH_GROUPS] = group_params_candidate
                
                candidate_orders.append(order)
                candidate_params.append(eval_params_candidate)

            results = model_eval.model_evaluation.evaluate_models(
                        self.data, config_dicts=candidate_params, enable_parallel=enable_parallel, 
                        workspace=parallel_workspace, time_limit="00:20:00", mem_limit=2, conda_env="amptorch")

            for i in range(len(candidate_orders)):
                curr_test_mse = results[i].test_error
                curr_order = candidate_orders[i]
                curr_params = candidate_params[i]

                if curr_test_mse < curr_min_test_mse:
                    curr_min_test_mse = curr_test_mse
                    curr_best_order = curr_order
                    curr_best_group_params = copy.deepcopy(curr_params[model_eval.constants.CONFIG_MCSH_GROUPS])

            max_change_pct = (curr_min_test_mse - prev_test_mse) / prev_test_mse
            print("Best change: removing order {} changed test MSE by {} pct ({} to {})".format(
                curr_best_order, max_change_pct, prev_test_mse, curr_min_test_mse))
            print("Params for best change: {}".format(curr_best_group_params))

            #check for stop criteria
            if max_change_pct < stop_change_pct:
                
                if max_change_pct < 0.:
                    prev_test_mse = curr_min_test_mse
        
                prev_group_params = copy.deepcopy(curr_best_group_params)

                MSEs.append(curr_min_test_mse)
                orders_removed.append(curr_best_order)

                if output_dir:
                    #write results to file (overwrite on each iteration)
                    results = pd.DataFrame(data={"order": orders_removed, 
                                                 "test_mse": MSEs, 
                                                 "iteration": range(len(MSEs))})
                    results.to_csv(os.path.join(output_dir, "backward_elimination_results.csv"))

            else:
                print("Best change was less than {} pct, stopping".format(stop_change_pct))
                break

        self.stats = pd.DataFrame(data={"order": orders_removed, 
                                        "test_mse": MSEs,
                                        "iteration": range(len(MSEs))})