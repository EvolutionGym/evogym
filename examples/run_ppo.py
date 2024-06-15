import os
import shutil
import json
import argparse
import numpy as np
from stable_baselines3 import PPO
import evogym.envs
from evogym import WorldObject

from ppo.args import add_ppo_args
from ppo.run import run_ppo
from ppo.eval import eval_policy
    
if __name__ == "__main__":    
    
    # Args
    parser = argparse.ArgumentParser(description='Arguments for PPO script')
    parser.add_argument(
        "--env-name", default="Walker-v0", type=str, help="Environment name (default: Walker-v0)"
    )
    parser.add_argument(
        "--save-dir", default="saved_data", type=str, help="Parent directory in which to save data(default: saved_data)"
    )
    parser.add_argument(
        "--exp-name", default="test_ppo", type=str, help="Name of experiment. Data saved to <save-dir/exp-name> (default: test_ppo)"
    )
    parser.add_argument(
        "--robot-path", default=os.path.join("world_data", "speed_bot.json"), type=str, help="Path to the robot json file (default: world_data/speed_bot.json)"
    )
    add_ppo_args(parser)
    args = parser.parse_args()
    
    # Manage dirs
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    if os.path.exists(exp_dir):
        print(f'THIS EXPERIMENT ({args.exp_name}) ALREADY EXISTS')
        print("Delete and override? (y/n): ", end="")
        ans = input()
        if ans.lower() != "y":
            exit()
        shutil.rmtree(exp_dir)
    model_save_dir = os.path.join(args.save_dir, args.exp_name, "controller")
    structure_save_dir = os.path.join(args.save_dir, args.exp_name, "structure")
    save_name = f"{args.env_name}"

    # Get Robot
    robot = WorldObject.from_json(args.robot_path)
    os.makedirs(structure_save_dir, exist_ok=True)
    np.savez(os.path.join(structure_save_dir, save_name), robot.get_structure(), robot.get_connections())

    # Train
    best_reward = run_ppo(
        args=args,
        body=robot.get_structure(),
        connections=robot.get_connections(),
        env_name=args.env_name,
        model_save_dir=model_save_dir,
        model_save_name=save_name,
    )
    
    # Save result file
    with open(os.path.join(args.save_dir, args.exp_name, "ppo_result.json"), "w") as f:
        json.dump({
            "best_reward": best_reward,
            "env_name": args.env_name,
        }, f, indent=4)

    # Evaluate
    model = PPO.load(os.path.join(model_save_dir, save_name))
    rewards = eval_policy(
        model=model,
        body=robot.get_structure(),
        connections=robot.get_connections(),
        env_name=args.env_name,
        n_evals=1,
        n_envs=1,
        render_mode="human",
    )
    print(f"Mean reward: {np.mean(rewards)}")