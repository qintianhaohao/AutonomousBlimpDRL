import os
import pickle
import json
import numpy as np
import ray
import sys
sys.path.append('/home/ros/catkin_ws/src/AutonomousBlimpDRL/RL')
sys.path.append('/home/ros/catkin_ws/src/AutonomousBlimpDRL/blimp_env')
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
sys.path.append('/home/ros/catkin_ws/devel/lib/python3/dist-packages')
import rl.rllib_script.agent.model.ray_model

from ray.rllib.algorithms import sac
from ray.tune.logger import pretty_print
from ray.tune.trainable.util import TrainableUtil


checkpoint_path = os.path.expanduser(
    "~/ray_results/ResidualPlanarNavigateEnv_SAC_test/"
    "SAC_ResidualPlanarNavigateEnv_2e983_00000_0_2022-11-23_07-12-49/checkpoint_001320"
)

auto_start_simulation = True  # start simulation
duration = int(0.5 * 3600 * 10 * 7) + 24193600

num_workers = 1

real_experiment = True  # no reset
evaluation_mode = False  # fix robotid, don't support multiworker

run_pid = False
windspeed = 0
buoyancy = 0.93
traj = "square"

trigger_dist = 7
init_alt = 100

# from blimp_env.envs.planar_navigate_env import ResidualPlanarNavigateEnv
from blimp_env.envs import ResidualPlanarNavigateEnv
ENV = ResidualPlanarNavigateEnv
# ENV = "residual_planar_navigate-v0"

# load config from params.json
run_base_dir = os.path.dirname(checkpoint_path)
config_path = os.path.join(run_base_dir, "params.json")
# config_path = os.path.join(run_base_dir, "params.pkl")
with open(config_path, "rb") as f:
    config = json.load(f)
    # config = pickle.load(f)

if run_pid:
    beta = 0.0
    disable_servo = True
else:
    beta = 0.5
    disable_servo = False


env_config = config["env_config"]
env_config.update(
    {
        "DBG": False,
        "evaluation_mode": evaluation_mode,
        "real_experiment": real_experiment,
        "seed": 123,
        "duration": duration,
        "beta": beta,
        "success_threshhold": trigger_dist,  # [meters]
    }
)
env_config["simulation"].update(
    {
        "gui": False,
        "auto_start_simulation": auto_start_simulation,
        "enable_meshes": True,
        "enable_wind": False,
        "enable_wind_sampling": False,
        "wind_speed": windspeed,
        "wind_direction": (1, 0),
        "enable_buoyancy_sampling": False,
        "buoyancy_range": [buoyancy, buoyancy],
        "position": (0, 0, init_alt),
    }
)

obs_config = {
    "noise_stdv": 0.05,
}
if "observation" in env_config:
    env_config["observation"].update(obs_config)
else:
    env_config["observation"] = obs_config

act_config = {
    "act_noise_stdv": 0.5,
    "disable_servo": disable_servo,
}
if "action" in env_config:
    env_config["action"].update(act_config)
else:
    env_config["action"] = act_config

square = [
    (40, 40, -init_alt, 3),
    (40, -40, -init_alt, 3),
    (-40, -40, -init_alt, 3),
    (-40, 40, -init_alt, 3),
]

target_config = {
    "type": "MultiGoal",
    "target_name_space": "goal_",
    "trigger_dist": trigger_dist,
    "wp_list": square,
    "enable_random_goal": False,
}
if "target" in env_config:
    env_config["target"].update(target_config)
else:
    env_config["target"] = target_config

config.update(
    {
        "create_env_on_driver": False,
        "num_workers": num_workers,
        "num_gpus": 1,
        "explore": False,
        "env_config": env_config,
        "horizon": 400,
        "rollout_fragment_length": 400,
        # "train_batch_size": 5600,
        # "sgd_minibatch_size": 512,
        "lr": 0,
        # "lr_schedule": None,
        # "num_sgd_iter": 0,
    }
)

# delete illegal key in config
# print(config)
# print(type(config))
remove_keys = ['record_env', 'callbacks', 'sample_collector']
for k in list(config.keys()):
    if k in remove_keys:
        del config[k]
    # if k == 'input_evaluation':
    #     del config[k]
    #     config["evaluation_config"]["off_policy_estimation_methods"]={'is': {'type': 'is'}, 'wis': {'type': 'wis'}}

# rewrite .metadata file to fit ray rllib v2.1
# metadata = TrainableUtil.load_metadata(checkpoint_path)
# metadata['saved_as_dict'] = True
# TrainableUtil.write_metadata(checkpoint_path, metadata)
# metadata = TrainableUtil.load_metadata(checkpoint_path)

# evaluate
ray.shutdown()
ray.init()
agent = sac.SAC(config=config, env=ENV)
agent.restore(checkpoint_path)
for _ in range(int(duration)):
    result = agent.train()
    # print(pretty_print(result))
    if result["timesteps_total"] >= duration:
        break
print("done")
