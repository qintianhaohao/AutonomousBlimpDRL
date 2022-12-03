import argparse
import os
import ray
import rl.rllib_script.agent.model.ray_model
# from blimp_env.blimp_env.envs import ResidualPlanarNavigateEnv
from blimp_env.envs import ResidualPlanarNavigateEnv
from blimp_env.envs.script import close_simulation
from ray import tune
# from ray.rllib.agents import ppo
from ray.rllib.algorithms import sac
from ray.tune.registry import register_env
from rl.rllib_script.util import find_nearest_power_of_two

# exp setup
ENV = ResidualPlanarNavigateEnv
AGENT = sac
AGENT_NAME = "SAC"
exp_name_posfix = "test"

restore = None
# restore = os.path.expanduser(
#     '~/ray_results/ResidualPlanarNavigateEnv_SAC_test/'
#     'SAC_ResidualPlanarNavigateEnv_2e983_00000_0_2022-11-23_07-12-49/checkpoint_001320'
# )

parser = argparse.ArgumentParser()
parser.add_argument("--gui", type=bool, default=False, help="Start with gazebo gui")
parser.add_argument("--num_gpus", type=bool, default=1, help="Number of gpu to use")
parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use")
parser.add_argument("--resume", type=bool, default=False, help="resume the last experiment")


def env_creator(env_config):
    return ENV(env_config)

if __name__ == "__main__":
    env_name = ENV.__name__
    agent_name = AGENT_NAME
    exp_name = env_name + "_" + agent_name + "_" + exp_name_posfix

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init(local_mode=False)
    register_env(env_name, env_creator)

    init_alt = 100
    square = [
        (40, 40, -init_alt, 3),
        (40, -40, -init_alt, 3),
        (-40, -40, -init_alt, 3),
        (-40, 40, -init_alt, 3),
    ]
    env_config = {
        # "seed": tune.grid_search([123, 456, 789]),
        "seed": 123,
        "simulation": {
            "gui": args.gui,
            "auto_start_simulation": True,
            "enable_wind": True,
            "enable_wind_sampling": True,
            "enable_buoyancy_sampling": True,
        },
        # "target": {
        #     "type": "MultiGoal",
        #     "target_name_space": "goal_",
        #     "trigger_dist": 5,
        #     "wp_list": square,
        #     "enable_random_goal": False,
        # },
        "mixer_type": "absolute",
        # alpha, beta,
        # set beta == 1 means only use rl control
        # set beta == 0 means only use pid control
        "mixer_param": (0.5, 1),
        "enable_residual_ctrl": False,

        "duration": 900,
        "simulation_frequency": 30,  # [hz]
        "policy_frequency": 10,  # [hz] has to be greater than 5 to overwrite backup controller
        "repeat_action_n_step": 10,
    }

    days = 2
    one_day_ts = 24 * 3600 * env_config['policy_frequency']
    TIMESTEP = int(days * one_day_ts)
    stop = {
        "timesteps_total": TIMESTEP,
    }
    # episode_ts must be integer
    episode_ts = env_config['duration'] * env_config['policy_frequency'] / env_config['simulation_frequency']
    train_batch_size = args.num_workers * 4 * episode_ts

    config = AGENT.DEFAULT_CONFIG.copy()
    config.update(
        {
            "env": env_name,
            "env_config": env_config,
            "log_level": "INFO",
            "num_gpus": args.num_gpus,
            "num_workers": args.num_workers,  # parallelism
            "num_envs_per_worker": 1,
            "framework": "torch",
            # == Learning ==
            "gamma": 0.999,
            "horizon": env_config['duration'],
            "rollout_fragment_length": episode_ts,
            "train_batch_size": train_batch_size,
            "lr": 1e-4,
            "grad_clip": 1.0,
            "observation_filter": "NoFilter",
            "batch_mode": "truncate_episodes",
            "replay_buffer_config": {
                "capacity": int(1e5),
                "prioritized_replay": True, # If True prioritized replay buffer will be used.
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                # Whether to compute priorities already on the remote worker side.
                "worker_side_prioritization": False,
            }
        }
    )

    print('---------- train config ----------')
    print(config)
    if env_config["simulation"]["auto_start_simulation"]:
        close_simulation()

    try:
        results = tune.run(
            AGENT_NAME,
            name=exp_name,
            config=config,
            stop=stop,
            checkpoint_freq=50,
            checkpoint_at_end=True,
            reuse_actors=False,
            restore=restore,
            resume=args.resume,
            max_failures=3,
            verbose=1,
        )
    except:
        raise ValueError

    ray.shutdown()
