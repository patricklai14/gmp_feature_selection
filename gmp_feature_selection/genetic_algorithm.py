import numpy as np
import pandas as pd

import copy
import os
import pathlib
import pdb

import model_eval
import gmp_feature_selection as gmp_fs

class gene_sequence:
    def __init__(self, gmp_groups, sigmas):
        self.groups = gmp_groups
        self.sigmas = sigmas

class genetic_algorithm(gmp_fs.gmp_feature_selector):
    def __init__(self, data, model_eval_params):
        super().__init__(data, model_eval_params)

    def generate_initial_population(self, population_size, target_num_groups, num_sigmas):
        #TODO: make this configurable
        #sigmas = [0.25, 1.0, 2.0]
        self.max_sigma = 2.5

        population = []
        for i in range(population_size):
            curr_indices = np.random.choice(self.seq_len, target_num_groups, replace=False)
            curr_groups = [False] * self.seq_len

            for idx in curr_indices:
                curr_groups[idx] = True

            curr_sigmas = np.random.uniform(0.0, self.max_sigma, size=num_sigmas)
            curr_sigmas = np.sort(curr_sigmas).tolist()

            curr_seq = gene_sequence(curr_groups, curr_sigmas)
            population.append(curr_seq)

        return population

    #convert gene representation to model parameters
    def get_model_eval_params_from_genes(self, sequences):
        eval_params = []
        for seq_num in range(len(sequences)):
            seq = sequences[seq_num]
            curr_eval_params = copy.deepcopy(self.model_eval_params)

            #extract gmp groups from gene encoding
            curr_gmp_groups = {}
            for i in range(len(seq.groups)):
                if seq.groups[i]:
                    #add corresponding group
                    curr_order, curr_group = self.all_groups[i]

                    if str(curr_order) not in curr_gmp_groups:
                        curr_gmp_groups[str(curr_order)] = {"groups": [], "sigmas": seq.sigmas}

                    curr_gmp_groups[str(curr_order)]["groups"].append(curr_group)

            curr_eval_params["name"] += "_seq_{}".format(seq_num) 
            curr_eval_params["amptorch_config"]["dataset"]["fp_params"]["MCSHs"] = curr_gmp_groups
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

    #perform uniform crossover
    def crossover(self, parents, target_num_groups):
        #gmp groups
        groups_1 = []
        groups_2 = []

        gene_origin_1 = np.random.choice([0, 1], size=self.seq_len, replace=True)
        gene_origin_2 = 1 - gene_origin_1
        for j in range(self.seq_len):
            groups_1.append(parents[gene_origin_1[j]].groups[j])
            groups_2.append(parents[gene_origin_2[j]].groups[j])

        #readjust sequence so that offspring have target number of groups
        num_groups_1 = np.sum(groups_1)
        if num_groups_1 != target_num_groups:
            num_swaps = abs(num_groups_1 - target_num_groups)

            if num_groups_1 < target_num_groups:
                possible_swaps = (~np.array(groups_1) & np.array(groups_2)).nonzero()[0]
            else:
                possible_swaps = (np.array(groups_1) & ~np.array(groups_2)).nonzero()[0]

            swap_indices = np.random.choice(possible_swaps, size=num_swaps, replace=False)
            for swap_idx in swap_indices:
                tmp = groups_1[swap_idx]
                groups_1[swap_idx] = groups_2[swap_idx]
                groups_2[swap_idx] = tmp

        #sigmas (arithmetic crossover)
        #make this configurable?
        delta = 0.25
        sigmas_parent_1 = np.array(parents[0].sigmas)
        sigmas_parent_2 = np.array(parents[1].sigmas)

        sigmas_1 = (delta * sigmas_parent_1) + ((1. - delta) * sigmas_parent_2) 
        sigmas_2 = (delta * sigmas_parent_2) + ((1. - delta) * sigmas_parent_1) 

        return (gene_sequence(groups_1, sigmas_1.tolist()), gene_sequence(groups_2, sigmas_2.tolist()))


    def mutate(self, sequence):
        #reverse sequence mutation for groups
        bounds = np.random.randint(0, len(sequence.groups), size=2)
        i = bounds.min()
        j = bounds.max()

        while i < j:
            tmp = sequence.groups[i]
            sequence.groups[i] = sequence.groups[j]
            sequence.groups[j] = tmp
            i += 1
            j -= 1

        sigma_change_range = self.max_sigma / len(sequence.sigmas)
        sigma_changes = np.random.uniform(-0.5 * sigma_change_range, 0.5 * sigma_change_range, len(sequence.sigmas))
        new_sigmas = np.sort(np.array(sequence.sigmas) + sigma_changes)
        sequence.sigmas = new_sigmas.tolist()

    def run(self, population_size, num_generations, target_groups_pct, max_order, num_sigmas=5, crossover_prob=0.75, 
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
        population = self.generate_initial_population(population_size, target_num_groups, num_sigmas)

        #compute fitness of initial population
        eval_params = self.get_model_eval_params_from_genes(population)
        results = model_eval.model_evaluation.evaluate_models(
                    dataset=self.data, config_dicts=eval_params, enable_parallel=enable_parallel,
                    workspace=parallel_workspace, time_limit=time_limit, mem_limit=mem_limit, conda_env=conda_env)
        errors = [metrics.test_error for metrics in results]
        fitness_scores = self.get_fitness_scores(errors)
        print("MAE for initial population: {}".format(errors))
        print("Fitness scores for initial population: {}".format(fitness_scores))

        #determine number of 
        if population_size % 2 == 0 or elitist:
            num_parent_pairs = population_size // 2
        else:
            num_parent_pairs = (population_size // 2) + 1

        for generation in range(num_generations):
            print("Running genetic algorithm generation {}".format(generation + 1))
            new_population = []

            #perform elitist selection
            if elitist:
                best_seq_idx = np.argmax(fitness_scores)
                new_population.append(copy.deepcopy(population[best_seq_idx]))

            #calculate selection probabilities and perform selection (for roulette wheel selection)
            print("Selecting parents for crossover")
            total_score = np.sum(fitness_scores)
            selection_probs = fitness_scores / total_score
            parents = zip(np.random.choice(population_size, size=num_parent_pairs, replace=True, p=selection_probs), 
                          np.random.choice(population_size, size=num_parent_pairs, replace=True, p=selection_probs))
            parents = [p for p in parents]
            print("parents: {}".format(parents))

            for i in range(num_parent_pairs):
                curr_parents = (population[parents[i][0]], population[parents[i][1]])

                #determine whether to perform crossover/mutation
                do_crossover = np.random.rand() < crossover_prob
                do_mutation_1 = np.random.rand() < mutation_prob
                do_mutation_2 = np.random.rand() < mutation_prob

                if do_crossover:
                    offspring_1, offspring_2 = self.crossover(curr_parents, target_num_groups)
                    
                else:
                    offspring_1 = copy.deepcopy(curr_parents[0])
                    offspring_2 = copy.deepcopy(curr_parents[1])

                #apply mutation(s) (reverse sequence)
                if do_mutation_1:
                    print("Mutating offspring 1 of parent pair {}".format(i))
                    self.mutate(offspring_1)
                if do_mutation_2:
                    print("Mutating offspring 2 of parent pair {}".format(i))
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
            results = model_eval.model_evaluation.evaluate_models(
                        dataset=self.data, config_dicts=eval_params, enable_parallel=enable_parallel,
                        workspace=parallel_workspace, time_limit=time_limit, mem_limit=mem_limit, conda_env=conda_env)
            errors = [metrics.test_error for metrics in results]
            fitness_scores = self.get_fitness_scores(errors)
            print("MAE for generation {}: {}".format(generation + 1, errors))
            print("Fitness scores for generation {}: {}".format(generation + 1, fitness_scores))

        #set best parameters
        best_idx = np.argmax(fitness_scores)
        print("Best params: {}".format(eval_params[best_idx]))
        self.best_error = errors[best_idx]
        self.best_params = eval_params[best_idx]
