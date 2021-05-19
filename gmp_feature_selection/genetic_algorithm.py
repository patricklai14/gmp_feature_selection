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

    def generate_initial_population(self, population_size, target_num_groups):
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

    #compute fitness scores from MAEs
    def get_fitness_scores(self, errors):
        errors_inverted = 1. / np.array(errors)

        #normalize
        min_inverted_error = np.min(errors_inverted)
        inverted_error_range = np.max(errors_inverted) - min_inverted_error
        fitness_scores = (errors_inverted - min_inverted_error) / inverted_error_range

        return fitness_scores

    #reverse sequence mutation
    def mutate(self, sequence):
        bounds = np.random.randint(0, len(sequence), size=2)
        i = bounds.min()
        j = bounds.max()

        while i < j:
            tmp = sequence[i]
            sequence[i] = sequence[j]
            sequence[j] = tmp

    def run(self, population_size, num_generations, target_groups_pct, max_order, crossover_prob=0.75, 
            mutation_prob=0.2, elitist=True, enable_parallel=True, parallel_workspace=None, time_limit="00:30:00", 
            mem_limit=2, conda_env="amptorch", seed=1):
        print("Running genetic algorithm with population_size={}, num_generations={}, target_groups_pct={}, "
              "max_order={}, crossover_prob={}, mutation_prob={}, elitist={}".format(
                population_size, num_generations, target_groups_pct, max_order, crossover_prob, mutation_prob, 
                elitist))

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
        target_num_groups = int(target_groups_pct * len(self.all_groups))
        population = self.generate_initial_population(population_size, target_num_groups)

        #compute fitness of initial population
        eval_params = self.get_model_eval_params_from_genes(population)
        errors = model_eval.model_evaluation.evaluate_models(
                    dataset=self.data, config_dicts=eval_params, enable_parallel=enable_parallel,
                    workspace=parallel_workspace, time_limit=time_limit, mem_limit=mem_limit, conda_env=conda_env)
        fitness_scores = self.get_fitness_scores(errors)
        print("Fitness scores for initial population: {}".format(fitness))

        #determine number of 
        if population_size % 2 == 0 or elitist:
            num_parent_pairs = population_size / 2
        else:
            num_parent_pairs = (population_size / 2) + 1

        for generation in range(num_generations):
            print("Running genetic algorithm generation {}".format(generation + 1))
            new_population = []

            #perform elitist selection
            if elitist:
                best_seq_idx = np.argmax(fitness_scores)
                new_population.append(population[best_seq_idx])

            #calculate selection probabilities and perform selection (for roulette wheel selection)
            print("Selecting parents for crossover")
            total_score = np.sum(fitness_scores)
            selection_probs = fitness_scores / total_score
            parents = zip(np.random.choice(population_size, size=num_parent_pairs, replace=True, p=selection_probs), 
                          np.random.choice(population_size, size=num_parent_pairs, replace=True, p=selection_probs))
            print("parents: {}".format(parents))

            for i in range(num_parent_pairs):
                offspring_1 = []
                offspring_2 = []

                #determine whether to perform crossover/mutation
                do_crossover = np.random.rand() < crossover_prob
                do_mutation_1 = np.random.rand() < mutation_prob
                do_mutation_2 = np.random.rand() < mutation_prob

                if do_crossover:
                    #perform crossover (uniform)
                    gene_origin = np.random.choice([0, 1], size=self.seq_len, replace=True)
                    for j in range(len(gene_origin)):
                        offspring.append(parents[i][gene_origin[j]])
                else:
                    offspring_1 = parents[0]
                    offspring_2 = parents[1]

                #apply mutation(s) (reverse sequence)
                if do_mutation_1:
                    self.mutate(offspring_1)
                if do_mutation_2:
                    self.mutate(offspring_2)

                new_population.append(offspring_1)
                new_population.append(offspring_2)

            #remove excess sequences. This could be necessary due to elitist selection/odd population size
            if len(new_population) > population_size:
                new_population = new_population[:population_size]

            #replace old population with new
            population = new_population

            #evaluate population
            eval_params = self.get_model_eval_params_from_genes(population)
            errors = model_eval.model_evaluation.evaluate_models(
                        dataset=self.data, config_dicts=eval_params, enable_parallel=enable_parallel,
                        workspace=parallel_workspace, time_limit=time_limit, mem_limit=mem_limit, conda_env=conda_env)
            fitness_scores = self.get_fitness_scores(errors)
            print("Fitness scores for generation {}: {}".format(generation + 1, fitness))

        #set best parameters
        best_idx = np.argmax(fitness_scores)
        print("Best params: {}".format(eval_params[best_idx]))
        self.best_error = errors[best_idx]
        self.best_params = eval_params[best_idx]
