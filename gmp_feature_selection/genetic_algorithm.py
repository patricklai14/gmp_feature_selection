import numpy as np
import pandas as pd

import copy
import os
import pathlib
import pdb

import model_eval
import gmp_feature_selection as gmp_fs

class genetic_algorithm(gmp_fs.gmp_feature_selector):
    def __init__(self, data, model_eval_params):
        super().__init__(data, model_eval_params)

    def generate_initial_population(self, target_num_groups):
        population = []
        for i in range(population_size):
            curr_indices = np.random.choice(self.seq_len, target_num_groups, replace=False)
            curr_genes = [False] * self.seq_len

            for idx in curr_indices:
                curr_genes[idx] = True

            population.append(curr_genes)

        return population

    #convert gene representation to model parameters
    def get_model_eval_params_from_genes(self, sequences):
        #TODO: make this configurable
        sigmas = [0.25, 1.0, 2.0]

        eval_params = []
        for seq in sequences:
            curr_eval_params = copy.deepcopy(self.model_eval_params)

            #extract gmp groups from gene encoding
            curr_gmp_groups = {}
            for i in range(len(seq)):
                if seq[i]:
                    #add corresponding group
                    curr_order, curr_group = self.all_groups[i]

                    if str(curr_order) not in curr_gmp_groups:
                        curr_gmp_groups[str(curr_order)] = {"groups": [], "sigmas": sigmas}

                    curr_gmp_groups[str(curr_order)]["groups"].append(curr_group)

            curr_eval_params["amptorch_config"]["dataset"]["fp_params"]["MCSHs"] = curr_mcsh_groups
            eval_params.append(curr_eval_params)

        return eval_params

    def run(self, population_size, num_generations, target_groups_pct, max_order, enable_parallel=True, 
            parallel_workspace=None, time_limit="00:30:00", mem_limit=2, conda_env="amptorch", seed=1):
        print("Running genetic algorithm with population_size={}, num_generations={}, target_groups_pct={}, "
              "max_order={}".format(
                population_size, num_generations, target_groups_pct, max_order))

        np.random.seed(seed)
        self.model_eval_params["amptorch_config"]["cmd"]["seed"] = seed

        #set up gene structure
        self.all_groups = []
        for order, groups in gmp_fs.groups_by_order.items():
            if order > max_order:
                continue

            for group in groups:
                self.all_groups.append((order, group))

        #length of gene sequence
        self.seq_len = len(self.all_groups)

        #generate initial population
        target_num_groups = int(target_groups_pct * len(group_pairs))
        population = self.generate_initial_population(target_num_groups)

        #compute fitness of initial population
        eval_params = self.get_model_eval_params_from_genes(population)
        scores = model_eval.model_evaluation.evaluate_models(
                    dataset=self.data, config_dicts=eval_params, enable_parallel=enable_parallel,
                    workspace=parallel_workspace, time_limit=time_limit, mem_limit=mem_limit, conda_env=conda_env)

        for i in range(num_generations):
            #perform crossover

            #apply mutation

            #replace old population with new