from ppo.group_ppo import run_group_ppo, SimJob

if __name__ == "__main__":

    run_group_ppo(
        experiment_name = 'test_group_ppo',
        sim_jobs = [
            SimJob(
                name = 'big',
                robots = ['speed_bot', 'carry_bot'],
                envs = ['Walker-v0', 'Carrier-v0'],
                train_iters = 50
            ),
            SimJob(
                name = 'small',
                robots = ['speed_bot'],
                envs = ['Walker-v0'],
                train_iters = 50
            )
        ]
    )