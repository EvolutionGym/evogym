import os
import numpy as np
import json
import shutil
import argparse

from ppo.run import run_ppo
from ppo.args import add_ppo_args
import utils.mp_group as mp
import evogym.envs
from evogym import WorldObject

class SimJob():

    def __init__(self, name, robots, envs):
        self.name = name
        self.robots = robots
        self.envs = envs

    def get_data(self,):
        return {'robots': self.robots, 'envs': self.envs}

class RunData():

    def __init__(self, robot, env, job_name):
        self.robot = robot
        self.env = env
        self.job_name = job_name
        self.reward = 0
    def set_reward(self, reward):
        print(f'setting reward for {self.robot} in {self.env}... {reward}')
        self.reward = reward

def read_robot_from_file(file_name):
    possible_paths = [
        os.path.join(file_name),
        os.path.join(f'{file_name}.npz'),
        os.path.join(f'{file_name}.json'),
        os.path.join('world_data', file_name),
        os.path.join('world_data', f'{file_name}.npz'),
        os.path.join('world_data', f'{file_name}.json'),
    ]

    best_path = None
    for path in possible_paths:
        if os.path.exists(path):
            best_path = path
            break

    if best_path.endswith('json'):
        robot_object = WorldObject.from_json(best_path)
        return (robot_object.get_structure(), robot_object.get_connections())
    if best_path.endswith('npz'):
        structure_data = np.load(best_path)
        structure = []
        for key, value in structure_data.items():
            structure.append(value)
        return tuple(structure)
    return None

def clean_name(name):
    while name.find('/') != -1:
        name = name[name.find('/')+1:]
    while name.find('\\') != -1:
        name = name[name.find('\\')+1:]
    while name.find('.') != -1:
        name = name[:name.find('.')]
    return name

def run_group_ppo(experiment_name, sim_jobs): 
    ### ARGS ###
    parser = argparse.ArgumentParser(description='Arguments for group PPO script')
    add_ppo_args(parser)
    args = parser.parse_args()

    ### MANAGE DIRECTORIES ###
    exp_path = os.path.join("saved_data", experiment_name)
    try:
        os.makedirs(exp_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print("Delete and override? (y/n): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(exp_path)
            print()
        else:
            quit()

    ### RUN JOBS ###
    run_data = []
    group = mp.Group()
    out = {}

    for job in sim_jobs:

        out[job.name] = {}
        save_path_structure = os.path.join(exp_path, f'{job.name}', "structure")
        save_path_controller = os.path.join(exp_path, f'{job.name}', "controller")

        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass

        count = 0
        for env_name in job.envs:
            out[job.name][env_name] = {}
            for robot_name in job.robots: 
                out[job.name][env_name][robot_name] = 0
                
                run_data.append(RunData(robot_name, env_name, job.name))
                structure = read_robot_from_file(robot_name)
                
                temp_path = os.path.join(save_path_structure, f'{clean_name(robot_name)}_{env_name}.npz')
                np.savez(temp_path, structure[0], structure[1])
                
                ppo_args = (args, structure[0], env_name, save_path_controller, f'{clean_name(robot_name)}_{env_name}', structure[1])
                group.add_job(run_ppo, ppo_args, callback=run_data[-1].set_reward)
                
    group.run_jobs(2)

    ### SAVE RANKING TO FILE ##
    for data in run_data:
        out[data.job_name][data.env][data.robot] = data.reward

    out_file = os.path.join(exp_path, 'output.json')
    with open(out_file, 'w') as f:
        json.dump(out, f)