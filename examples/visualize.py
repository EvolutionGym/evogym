import os, sys
root_dir = os.path.dirname(os.path.abspath(__file__))
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'PyTorch-NEAT'))
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

import json
import argparse
import sys
import numpy as np
import torch
import gym

from utils.algo_utils import *
from ppo.envs import make_vec_envs
from ppo.utils import get_vec_normalize

import evogym.envs

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

            env = make_vec_envs(
                args.env_name,
                structure,
                1000,
                1,
                None,
                None,
                device='cpu',
                allow_early_resets=False)

            # We need to use the same statistics for normalization as used in training
            try:
                save_path_controller = os.path.join(EXPERIMENT_PARENT_DIR, exp_name, "generation_" + str(gen_number), "controller", "robot_" + str(robot_index) + "_controller" + ".pt")
                actor_critic, obs_rms = \
                            torch.load(save_path_controller,
                                        map_location='cpu')
            except:
                print(f'\nCould not load robot controller data at {save_path_controller}.\n')
                continue

            vec_norm = get_vec_normalize(env)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.obs_rms = obs_rms

            recurrent_hidden_states = torch.zeros(1,
                                                actor_critic.recurrent_hidden_state_size)
            masks = torch.zeros(1, 1)

            obs = env.reset()
            env.render('screen')

            total_steps = 0
            reward_sum = 0
            while total_steps < num_iters:
                with torch.no_grad():
                    value, action, _, recurrent_hidden_states = actor_critic.act(
                        obs, recurrent_hidden_states, masks, deterministic=args.det)


                # Obser reward and next obs
                obs, reward, done, _ = env.step(action)
                masks.fill_(0.0 if (done) else 1.0)
                reward_sum += reward
                
                if done == True:
                    env.reset()
                    reward_sum = float(reward_sum.numpy().flatten()[0])
                    print(f'\ntotal reward: {round(reward_sum, 5)}\n')
                    reward_sum = 0

                env.render('screen')

                total_steps += 1
            
            env.venv.close()

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

        env = make_vec_envs(
            env_name,
            structure,
            1000,
            1,
            None,
            None,
            device='cpu',
            allow_early_resets=False)

        # We need to use the same statistics for normalization as used in training
        try:
            save_path_controller = os.path.join(exp_dir, job, "controller", f"robot_{robot}_{env_name}_controller.pt")
            actor_critic, obs_rms = \
                        torch.load(save_path_controller,
                                    map_location='cpu')
        except:
            print(f'\nCould not load robot controller data at {save_path_controller}.\n')
            continue

        vec_norm = get_vec_normalize(env)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.obs_rms = obs_rms

        recurrent_hidden_states = torch.zeros(1,
                                            actor_critic.recurrent_hidden_state_size)
        masks = torch.zeros(1, 1)

        obs = env.reset()
        env.render('screen')

        total_steps = 0
        reward_sum = 0
        while total_steps < num_iters:
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=args.det)


            # Obser reward and next obs
            obs, reward, done, _ = env.step(action)
            masks.fill_(0.0 if (done) else 1.0)
            reward_sum += reward

            if done == True:
                env.reset()
                reward_sum = float(reward_sum.numpy().flatten()[0])
                print(f'\ntotal reward: {round(reward_sum, 5)}\n')
                reward_sum = 0

            env.render('screen')

            total_steps += 1
        
        env.venv.close()

EXPERIMENT_PARENT_DIR = os.path.join(root_dir, 'saved_data')
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
    # group ppo experiment
    if 'output.json' in files_in_exp_dir:
        visualize_group_ppo(args, exp_name)
    # codesign experiment
    else:
        visualize_codesign(args, exp_name)