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

    def run(self, sigmas, stop_change_pct=0.15, selection_type="groups", enable_parallel=False, 
            parallel_workspace=None, time_limit="00:30:00", mem_limit=2, conda_env="amptorch"):
        #setup baseline MCSH params
        base_order = 5 #number of orders to include by default
        base_groups = {str(i): {"groups": constants.groups_by_order[i], "sigmas": sigmas} 
                        for i in range(base_order + 1)}

        base_params = self.model_eval_params
        base_params["name"] = "base"
        base_params["amptorch_config"]["dataset"]["fp_scheme"] = "mcsh"
        base_params["amptorch_config"]["dataset"]["fp_params"]["MCSHs"] = base_groups

        #get baseline performance
        print("Testing base params: {}".format(base_params))
        base_results = model_eval.model_evaluation.evaluate_models(
                        dataset=self.data, config_dicts=[base_params], enable_parallel=True,
                        workspace=parallel_workspace, time_limit=time_limit, 
                        mem_limit=mem_limit, conda_env=conda_env)
        base_test_error = base_results[0].test_error
        print("Base test MSE: {}".format(base_test_error))

        prev_test_error = base_test_error
        prev_groups = copy.deepcopy(base_groups)

        errors = [base_test_error]
        removed = [-1]

        print("Backward elimination params: stop_change_pct={}".format(
                stop_change_pct))

        #perform backward elimination
        while True:
            curr_min_test_error = 1000000.
            curr_best_id = -1
            curr_best_groups = None

            candidate_id = [] #either removed group or order
            candidate_params = []
            
            print("Creating configs for processing on Pace")
            for order, order_params in prev_groups.items():
                for group in order_params["groups"]:
                    groups_candidate = copy.deepcopy(prev_groups)
                    
                    #remove group
                    groups_candidate[order]["groups"].remove(group)

                    #remove order if no groups remaining
                    if not groups_candidate[order]["groups"]:
                        del groups_candidate[order]

                    curr_id = (order, group)
                    eval_params_candidate = copy.deepcopy(base_params)
                    eval_params_candidate["name"] = "back_elim_{}_{}".format(order, group)
                    eval_params_candidate["amptorch_config"]["dataset"]["fp_params"]["MCSHs"] = groups_candidate
                    
                    candidate_id.append(curr_id)
                    candidate_params.append(eval_params_candidate)

            results = model_eval.model_evaluation.evaluate_models(
                        dataset=self.data, config_dicts=candidate_params, enable_parallel=enable_parallel, 
                        workspace=parallel_workspace, time_limit=time_limit, mem_limit=mem_limit, conda_env=conda_env)

            for i in range(len(candidate_id)):
                curr_test_error = results[i].test_error
                curr_id = candidate_id[i]
                curr_params = candidate_params[i]

                if curr_test_error < curr_min_test_error:
                    curr_min_test_error = curr_test_error
                    curr_best_id = curr_id
                    curr_best_groups = copy.deepcopy(curr_params["amptorch_config"]["dataset"]["fp_params"]["MCSHs"])
                    self.best_params = curr_params
                    self.best_error = curr_test_error

            max_change_pct = (curr_min_test_error - prev_test_error) / prev_test_error
            print("Best change: removing group {} changed test error by {} pct ({} to {})".format(
                curr_best_id, max_change_pct, prev_test_error, curr_min_test_error))
            print("Groups for best change: {}".format(curr_best_groups))

            #check for stop criteria
            if max_change_pct < stop_change_pct:
                if max_change_pct < 0.:
                    prev_test_error = curr_min_test_error
        
                prev_groups = copy.deepcopy(curr_best_groups)

                errors.append(curr_min_test_error)
                removed.append(curr_best_id)

            else:
                print("Best change was less than {} pct, stopping".format(stop_change_pct))
                break

        self.stats = {"groups": removed, 
                      "test_error": errors}
