import os

import json
import argparse
import numpy as np
from typing import Optional

from utils.algo_utils import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import evogym.envs

def rollout(
    env_name: str,
    n_iters: int,
    model: PPO,
    body: np.ndarray,
    connections: Optional[np.ndarray] = None,
    seed: int = 42,
):
    # Parallel environments
    vec_env = make_vec_env(env_name, n_envs=1, seed=seed, env_kwargs={
        'body': body,
        'connections': connections,
        "render_mode": "human",
    })
    
    # Rollout
    reward_sum = 0
    obs = vec_env.reset()
    count = 0
    while count < n_iters:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        reward_sum += reward[0]
        count += 1
        if done:
            print(f'\nTotal reward: {reward_sum:.5f}\n')
    vec_env.close()

def visualize_codesign(args, exp_name):
    global EXPERIMENT_PARENT_DIR
    gen_list = os.listdir(os.path.join(EXPERIMENT_PARENT_DIR, exp_name))

    assert args.env_name != None, (
        'Visualizing this experiment requires an environment be specified as a command line argument. Eg: --env-name "Walker-v0"'
    )

    gen_count = 0
    while gen_count < len(gen_list):
        try:
            gen_list[gen_count] = int(gen_list[gen_count].split("_")[1])
        except:
            del gen_list[gen_count]
            gen_count -= 1
        gen_count += 1

    all_robots = []
    if "cppn" in exp_name:
        for gen in gen_list:
            gen_data_path = os.path.join(EXPERIMENT_PARENT_DIR, exp_name, "generation_" + str(gen), "output.txt")
            f = open(gen_data_path, "r")
            count = 1
            for line in f:
                all_robots.append((gen, count, float(line.split()[1])))
                #(f'{count} | {line.split()[1]} (ID: {line.split()[0]})')
                count += 1
    all_robots = sorted(all_robots, key=lambda x: x[2], reverse=True)
    num_robots_to_print_cppn = 30 if len(all_robots) > 10 else len(all_robots)

    while(True):

        if len(all_robots) > 0:
            print()
        for i in range(num_robots_to_print_cppn):
            print(f'gen: {all_robots[i][0]} |\t ind: {all_robots[i][1]}|\t r: {all_robots[i][2]}')

        pretty_print(sorted(gen_list))
        print()

        print("Enter generation number: ", end="")
        gen_number = int(input())

        gen_data_for_printing = []
        gen_data = []
        gen_data_path = os.path.join(EXPERIMENT_PARENT_DIR, exp_name, "generation_" + str(gen_number), "output.txt")
        f = open(gen_data_path, "r")
        count = 1
        for line in f:
            gen_data_for_printing.append(f'{count} | {line.split()[1]} (ID: {line.split()[0]})') 
            gen_data.append((line.split()[0], line.split()[1]))
            count += 1

        print()
        pretty_print(gen_data_for_printing)
        print()

        print("Enter robot rank: ", end="")
        robot_ranks = parse_range(input(), len(gen_data))

        print("Enter num iters: ", end="")
        num_iters = int(input())

        for robot_rank in robot_ranks:

            robot_index = gen_data[robot_rank-1][0]
            try:
                save_path_structure = os.path.join(EXPERIMENT_PARENT_DIR, exp_name, "generation_" + str(gen_number), "structure", str(robot_index) + ".npz")
                structure_data = np.load(save_path_structure)
                structure = []
                for key, value in structure_data.items():
                    structure.append(value)
                structure = tuple(structure)
                print(f'\nStructure for rank {robot_rank} robot (index {robot_index}):\n{structure}\n')
            except:
                print(f'\nCould not load robot strucure data at {save_path_structure}.\n')
                continue

            if num_iters == 0:
                continue
            
            save_path_controller = os.path.join(EXPERIMENT_PARENT_DIR, exp_name, "generation_" + str(gen_number), "controller", f'{robot_index}.zip')
            model = PPO.load(save_path_controller)
            rollout(args.env_name, num_iters, model, structure[0], structure[1])

def visualize_group_ppo(args, exp_name):

    exp_dir = os.path.join(EXPERIMENT_PARENT_DIR, exp_name)
    out_file = os.path.join(exp_dir, 'output.json')
    out = {}
    with open(out_file, 'r') as f:
        out = json.load(f)

    jobs = list(out.keys())
    jobs_p = []
    for i, job in enumerate(jobs):
        jobs_p.append(f'{job} ({i})')

    while True:
        pretty_print(jobs_p)
        print()

        print("Enter job number: ", end="")
        job_num = int(input())
        while(job_num < 0 or job_num >= len(jobs)):
            print("Enter job number: ", end="")
            job_num = int(input())

        job = jobs[job_num]
        job_data = out[job]

        robot_data = []
        robots_p = []
        for env in job_data.keys():
            for robot in job_data[env].keys():
                reward = job_data[env][robot]
                robot_data.append((env, reward, robot))
        robot_data = sorted(robot_data, reverse=True)

        for i, data in enumerate(robot_data):
            env_name, reward, robot = data
            robots_p.append(f'{env_name}, {robot}: {reward} | ({i})')
        
        pretty_print(robots_p, max_name_length=60)
        print()

        print("Enter sim number: ", end="")
        sim_num = int(input())
        while(sim_num < 0 or sim_num >= len(robot_data)):
            print("Enter sim number: ", end="")
            sim_num = int(input())

        env_name, reward, robot = robot_data[sim_num]

        print("Enter num iters: ", end="")
        num_iters = int(input())
        print()

        if num_iters == 0:
            continue

        save_path_structure = os.path.join(exp_dir, job, "structure", f"{robot}_{env_name}.npz")
        structure_data = np.load(save_path_structure)
        structure = []
        for key, value in structure_data.items():
            structure.append(value)
        structure = tuple(structure)
        
        save_path_controller = os.path.join(exp_dir, job, "controller", f"{robot}_{env_name}.zip")
        model = PPO.load(save_path_controller)
        rollout(env_name, num_iters, model, structure[0], structure[1])
        
def visualize_ppo(args, exp_name):

    exp_dir = os.path.join(EXPERIMENT_PARENT_DIR, exp_name)
    out_file = os.path.join(exp_dir, 'ppo_result.json')
    out = {}
    with open(out_file, 'r') as f:
        out = json.load(f)
        
    reward = out['best_reward']
    env_name = out['env_name']
    
    print(f'\nEnvironment: {env_name}\nReward: {reward}')

    while True:
        print()
        print("Enter num iters: ", end="")
        num_iters = int(input())
        print()

        if num_iters == 0:
            continue

        save_path_structure = os.path.join(exp_dir, "structure", f"{env_name}.npz")
        structure_data = np.load(save_path_structure)
        structure = []
        for key, value in structure_data.items():
            structure.append(value)
        structure = tuple(structure)
        
        save_path_controller = os.path.join(exp_dir, "controller", f"{env_name}.zip")
        model = PPO.load(save_path_controller)
        rollout(env_name, num_iters, model, structure[0], structure[1])

EXPERIMENT_PARENT_DIR = os.path.join('saved_data')
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--env-name',
        help='environment to train on')
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')
    args = parser.parse_args()
    args.det = not args.non_det

    exp_list = os.listdir(EXPERIMENT_PARENT_DIR)
    pretty_print(exp_list)

    print("\nEnter experiment name: ", end="")
    exp_name = input()
    while exp_name not in exp_list:
        print("Invalid name. Try again:")
        exp_name = input()

    files_in_exp_dir = os.listdir(os.path.join(EXPERIMENT_PARENT_DIR, exp_name))
    
    if 'output.json' in files_in_exp_dir: # group ppo experiment
        visualize_group_ppo(args, exp_name)
    elif 'ppo_result.json' in files_in_exp_dir: # ppo experiment
        visualize_ppo(args, exp_name)
    else: # codesign experiment
        visualize_codesign(args, exp_name)