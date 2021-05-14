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
        return []


    def run(self, population_size, num_generations, target_groups_pct, max_order, enable_parallel=True, 
            parallel_workspace=None, time_limit="00:30:00", mem_limit=2, conda_env="amptorch", seed=1):
        print("Running genetic algorithm with population_size={}, num_generations={}, target_groups_pct={}, "
              "max_order={}".format(
                population_size, num_generations, target_groups_pct, max_order))

        np.random.seed(seed)

        #set up gene structure
        num_groups = 0
        for order, groups in gmp_fs.groups_by_order.items():
            if order > max_order:
                continue

            num_groups += len(groups)

        #length of gene sequence
        self.seq_len = num_groups

        #generate initial population
        target_num_groups = int(target_groups_pct * len(group_pairs))
        population = self.generate_initial_population(target_num_groups)

        #compute fitness of initial population


        #loop
            #perform crossover

            #apply mutation

            #replace old population with new