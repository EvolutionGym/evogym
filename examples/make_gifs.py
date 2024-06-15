import os
import numpy as np
import torch
from PIL import Image
import imageio
from pygifsicle import optimize

import evogym.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from utils.algo_utils import *
import utils.mp_group as mp

def get_generations(load_dir, exp_name):
    gen_list = os.listdir(os.path.join(load_dir, exp_name))
    gen_count = 0
    while gen_count < len(gen_list):
        try:
            gen_list[gen_count] = int(gen_list[gen_count].split("_")[1])
        except:
            del gen_list[gen_count]
            gen_count -= 1
        gen_count += 1
    return [i for i in range(gen_count)]

def get_exp_gen_data(exp_name, load_dir, gen):
    robot_data = []
    gen_data_path = os.path.join(load_dir, exp_name, f"generation_{gen}", "output.txt")
    f = open(gen_data_path, "r")
    for line in f:
        robot_data.append((int(line.split()[0]), float(line.split()[1])))
    return robot_data

def save_robot_gif(out_path, env_name, body_path, ctrl_path, seed=42):
    global GIF_RESOLUTION

    structure_data = np.load(body_path)
    structure = []
    for key, value in structure_data.items():
        structure.append(value)
    structure = tuple(structure)
    
    model = PPO.load(ctrl_path)
    
    # Parallel environments
    vec_env = make_vec_env(env_name, n_envs=1, seed=seed, env_kwargs={
        'body': structure[0],
        'connections': structure[1],
        "render_mode": "img",
    })
    
    obs = vec_env.reset()
    imgs = [vec_env.env_method('render')[0]] # vec env is stupid; .render() dosent work
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        imgs.append(vec_env.env_method('render')[0])

    imageio.mimsave(f'{out_path}.gif', imgs, duration=(1/50.0))
    try:
        optimize(out_path)
    except:
        pass
        # print("Error optimizing gif. Most likely cause is that gifsicle is not installed.")
    return 0

class Robot():
    def __init__(
        self, 
        body_path=None, 
        ctrl_path=None, 
        reward=None, 
        env_name=None, 
        exp_name=None, 
        gen=None):
        self.body_path = body_path
        self.ctrl_path = ctrl_path
        self.reward = reward
        self.env_name = env_name
        self.exp_name = exp_name
        self.gen = gen
    def __str__(self):
        exp_str = f'{self.exp_name}' if self.exp_name is not None else ''
        gen_str = f'gen{self.gen}' if self.gen is not None else ''
        reward_str = f'({round(self.reward, 3)})' if self.reward is not None else ''
        comps = [exp_str, gen_str, reward_str]
        out = ''
        for comp in comps:
            if len(comp) != 0:
                out += f'{comp}_'
        return out[:-1]
    
class Job():
    def __init__(
        self, 
        name, 
        experiment_names, 
        env_names, 
        load_dir,
        generations=None, 
        ranks=None,
        jobs=None,
        organize_by_jobs=True,
        organize_by_experiment=False,
        organize_by_generation=False):

        # set values
        self.name = name
        self.experiment_names = experiment_names
        self.env_names = env_names
        self.load_dir = load_dir
        self.generations = generations
        self.ranks = ranks

        # set jobs 
        self.sub_jobs = []
        if jobs:
            for job in jobs:
                self.sub_jobs.append(job)
                self.sub_jobs[-1].name = job.name if organize_by_jobs else None
        if organize_by_experiment:
            for exp_name, env_name in zip(self.experiment_names, self.env_names):
                self.sub_jobs.append(Job(
                    name = exp_name,
                    experiment_names = [exp_name],
                    env_names = [env_names],
                    load_dir = self.load_dir,
                    generations = self.generations,
                    ranks = self.ranks,
                    organize_by_experiment=False,
                    organize_by_generation=organize_by_generation
                ))
            self.experiment_names = None
            self.env_names = None
            self.generations = None
            self.ranks = None
        elif organize_by_generation:
            assert len(self.experiment_names) == 1, (
                'Cannot create generation level folders for multiple experiments. Quick fix: set organize_by_experiment=True.'
            )
            if self.generations == None:
                exp_name = self.experiment_names[0]
                self.generations = get_generations(self.load_dir, exp_name)
            for gen in self.generations:
                self.sub_jobs.append(Job(
                    name = f'generation_{gen}',
                    experiment_names = self.experiment_names,
                    env_names = self.env_names,
                    load_dir = self.load_dir,
                    generations = [gen],
                    ranks = self.ranks,
                    organize_by_experiment=False,
                    organize_by_generation=False
                ))
            self.experiment_names = None
            self.env_names = None
            self.generations = None
            self.ranks = None

    def generate(self, load_dir, save_dir, depth=0):
        if self.name is not None and len(self.name) != 0:
            save_dir = os.path.join(save_dir, self.name)

        tabs = '  '*depth
        print(f"{tabs}\{self.name}")
    
        try: os.makedirs(save_dir)
        except: pass

        for sub_job in self.sub_jobs:
            sub_job.generate(load_dir, save_dir, depth+1)

        # collect robots
        if self.experiment_names == None:
            return 

        robots = []
        for exp_name, env_name in zip(self.experiment_names, self.env_names):
            exp_gens = self.generations if self.generations is not None else get_generations(self.load_dir, exp_name)
            for gen in exp_gens:
                for idx, reward in get_exp_gen_data(exp_name, load_dir, gen):
                    robots.append(Robot(
                        body_path = os.path.join(load_dir, exp_name, f"generation_{gen}", "structure", f"{idx}.npz"),
                        ctrl_path = os.path.join(load_dir, exp_name, f"generation_{gen}", "controller", f"{idx}.zip"),
                        reward = reward,
                        env_name = env_name,
                        exp_name = exp_name if len(self.experiment_names) != 1 else None,
                        gen = gen if len(exp_gens) != 1 else None,
                    ))

        # sort and generate
        robots = sorted(robots, key=lambda x: x.reward, reverse=True)
        ranks = self.ranks if self.ranks is not None else [i for i in range(len(robots))]
        
        # make gifs
        for i, robot in zip(ranks, robots):
            save_robot_gif(
                os.path.join(save_dir, f'{i}_{robot}'),
                robot.env_name,
                robot.body_path,
                robot.ctrl_path
            )
    
GIF_RESOLUTION = (1280/5, 720/5)
if __name__ == '__main__':
    exp_root = os.path.join('saved_data')
    save_dir = os.path.join('saved_data', 'all_media')

    my_job = Job(
        name = 'test_ga',
        experiment_names= ['test_ga'],
        env_names = ['Walker-v0'],
        ranks = [i for i in range(3)],
        load_dir = exp_root,
        organize_by_experiment=False,
        organize_by_generation=True,
    )
    
    my_job.generate(load_dir=exp_root, save_dir=save_dir)