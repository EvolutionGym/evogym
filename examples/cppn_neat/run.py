import os
import shutil
import random
import numpy as np
import torch
import neat

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'PyTorch-NEAT'))
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from pytorch_neat.cppn import create_cppn
from .parallel import ParallelEvaluator
from .population import Population

from utils.algo_utils import TerminationCondition
from ppo import run_ppo
from evogym import is_connected, has_actuator, get_full_connectivity, hashable
import evogym.envs


def get_cppn_input(structure_shape):
    x, y = torch.meshgrid(torch.arange(structure_shape[0]), torch.arange(structure_shape[1]))
    x, y = x.flatten(), y.flatten()
    center = (np.array(structure_shape) - 1) / 2
    d = ((x - center[0]) ** 2 + (y - center[1]) ** 2).sqrt()
    return x, y, d

def get_robot_from_genome(genome, config):
    nodes = create_cppn(genome, config, leaf_names=['x', 'y', 'd'], node_names=['empty', 'rigid', 'soft', 'hori', 'vert'])
    structure_shape = config.extra_info['structure_shape']
    x, y, d = get_cppn_input(structure_shape)
    material = []
    for node in nodes:
        material.append(node(x=x, y=y, d=d).numpy())
    material = np.vstack(material).argmax(axis=0)
    robot = material.reshape(structure_shape)
    return robot

def eval_genome_fitness(genome, config, genome_id, generation):
    robot = get_robot_from_genome(genome, config)
    connectivity = get_full_connectivity(robot)
    save_path_generation = os.path.join(config.extra_info['save_path'], f'generation_{generation}')
    save_path_structure = os.path.join(save_path_generation, 'structure', f'{genome_id}')
    save_path_controller = os.path.join(save_path_generation, 'controller')
    np.savez(save_path_structure, robot, connectivity)
    fitness = run_ppo(
        structure=(robot, connectivity),
        termination_condition=TerminationCondition(config.extra_info['train_iters']),
        saving_convention=(save_path_controller, genome_id),
    )
    return fitness

def eval_genome_constraint(genome, config, genome_id, generation):
    robot = get_robot_from_genome(genome, config)
    validity = is_connected(robot) and has_actuator(robot)
    if validity:
        robot_hash = hashable(robot)
        if robot_hash in config.extra_info['structure_hashes']:
            validity = False
        else:
            config.extra_info['structure_hashes'][robot_hash] = True
    return validity


class SaveResultReporter(neat.BaseReporter):

    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.generation = None

    def start_generation(self, generation):
        self.generation = generation
        save_path_structure = os.path.join(self.save_path, f'generation_{generation}', 'structure')
        save_path_controller = os.path.join(self.save_path, f'generation_{generation}', 'controller')
        os.makedirs(save_path_structure, exist_ok=True)
        os.makedirs(save_path_controller, exist_ok=True)

    def post_evaluate(self, config, population, species, best_genome):
        save_path_ranking = os.path.join(self.save_path, f'generation_{self.generation}', 'output.txt')
        genome_id_list, genome_list = np.arange(len(population)), np.array(list(population.values()))
        sorted_idx = sorted(genome_id_list, key=lambda i: genome_list[i].fitness, reverse=True)
        genome_id_list, genome_list = list(genome_id_list[sorted_idx]), list(genome_list[sorted_idx])
        with open(save_path_ranking, 'w') as f:
            out = ''
            for genome_id, genome in zip(genome_id_list, genome_list):
                out += f'{genome_id}\t\t{genome.fitness}\n'
            f.write(out)

def run_cppn_neat(
        experiment_name,
        structure_shape,
        pop_size,
        max_evaluations,
        train_iters,
        num_cores,
    ):

    save_path = os.path.join(root_dir, 'saved_data', experiment_name)

    try:
        os.makedirs(save_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print('Override? (y/n): ', end='')
        ans = input()
        if ans.lower() == 'y':
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            return None, None
        print()

    save_path_metadata = os.path.join(save_path, 'metadata.txt')
    with open(save_path_metadata, 'w') as f:
        f.write(f'POP_SIZE: {pop_size}\n' \
            f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n' \
            f'MAX_EVALUATIONS: {max_evaluations}\n' \
            f'TRAIN_ITERS: {train_iters}\n')

    structure_hashes = {}

    config_path = os.path.join(curr_dir, 'neat.cfg')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
        extra_info={
            'structure_shape': structure_shape,
            'train_iters': train_iters,
            'save_path': save_path,
            'structure_hashes': structure_hashes,
        },
        custom_config=[
            ('NEAT', 'pop_size', pop_size),
        ],
    )

    pop = Population(config)
    reporters = [
        neat.StatisticsReporter(),
        neat.StdOutReporter(True),
        SaveResultReporter(save_path),
    ]
    for reporter in reporters:
        pop.add_reporter(reporter)

    evaluator = ParallelEvaluator(num_cores, eval_genome_fitness, eval_genome_constraint)

    pop.run(
        evaluator.evaluate_fitness,
        evaluator.evaluate_constraint,
        n=np.ceil(max_evaluations / pop_size))

    best_robot = get_robot_from_genome(pop.best_genome, config)
    best_fitness = pop.best_genome.fitness
    return best_robot, best_fitness