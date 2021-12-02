import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models
import torch
import gym, sys
import cv2
from tqdm import tqdm
from IPython.display import clear_output
from pathlib import Path
from rlxai.agent.ga_cnn import ReinforceGA
from rlxai.dp.img import data_transform
from collections import Counter
from tqdm import tqdm

# env = gym.make("Breakout-v0")
env = gym.make("Breakout-v4")  #
# Deterministic

action_dim = env.action_space.n

############################################################
rewards = []  #
is_terminals = []  #
values = []  #
logprobs = []  #
############################################################

random_seed = 566
if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

model_path = Path("./model")
model_algo_path = model_path.joinpath("./GA")
model_algo_path.mkdir(exist_ok=True, parents=True)

import pygad.torchga
from pygad import torchga

device = "cpu"  ## only suport cpu
model = ReinforceGA(3, action_dim=action_dim, device=device)
try:
    print("load model")
    model.load(model_algo_path.joinpath("./model.pt"))
except Exception as e:
    print(e)

torch_ga = pygad.torchga.TorchGA(model=model.actor, num_solutions=10)


# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 10_000  # Number of generations.
num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights  # Initial population of network weights
parent_selection_type = "sss"  # Type of parent selection.
crossover_type = "single_point"  # Type of the crossover operator.
mutation_type = "random"  # Type of the mutation operator.
mutation_percent_genes = (
    10  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
)
keep_parents = (
    -1
)  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.


with tqdm(total=num_generations) as pbar:
    BEST_RESULT = 0

    def fitness_func(solution, sol_idx):
        global torch_ga, model, BEST_RESULT, model_algo_path
        model_weights_dict = torchga.model_weights_as_dict(model=model.actor, weights_vector=solution)
        model.actor.load_state_dict(model_weights_dict)
        env = gym.make("Breakout-v4")
        state = env.reset()
        state, _, _, info = env.step(1)
        state_list = [data_transform(state) for _ in range(3)]
        sum_reward = 0
        last_lives = info["lives"]
        while True:
            state = np.concatenate(state_list, axis=1) / 255.0
            action_dict = model.select_action(state)
            if last_lives > info["lives"]:
                action = 1
                last_lives = info["lives"]
            else:
                action = int(action_dict["action"])
            n_state, reward, done, info = env.step(action)
            state_list.pop(0)
            cur_lives = info["lives"]
            if last_lives > cur_lives:
                n_state = np.zeros_like(n_state)
            state_list.append(data_transform(n_state))
            sum_reward += reward
            if done:
                break
        if BEST_RESULT < sum_reward:
            BEST_RESULT = sum_reward
            model.save(model_algo_path.joinpath("./model.pt"))
        pbar.set_description(f"BEST : {int(BEST_RESULT):03d}")
        return sum_reward

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        initial_population=initial_population,
        fitness_func=fitness_func,
        parent_selection_type=parent_selection_type,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        keep_parents=keep_parents,
        on_generation=lambda _: pbar.update(1),
        stop_criteria=["saturate_1000"],
    )
    ga_instance.run()

ga_instance.plot_result(
    save_dir=model_algo_path.joinpath("ga_plot_result.png"), title="PyGAD & Torch - Iteration vs. Fitness", linewidth=4
)
ga_instance.plot_fitness(save_dir=model_algo_path.joinpath("ga_plot_fitness.png"), plot_type="plot")

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

model_weights_as_dict = pygad.torchga.model_weights_as_dict(model=model.actor, weights_vector=solution)
model.actor.load_state_dict(model_weights_as_dict)
model.save(model_algo_path.joinpath("./model.pt"))
