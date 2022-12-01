import copy
import argparse
import numpy as np
from blimp_env.envs.common.gazebo_connection import GazeboConnection
from blimp_env.envs.script import close_simulation
from blimp_env.envs import ResidualPlanarNavigateEnv


parser = argparse.ArgumentParser()
parser.add_argument(
    "--auto_start_simulation",
    dest="auto_start_simulation",
    default=False,
    action="store_true",
    help="spawn: ros core, gazebo, blimp model",
)
args = parser.parse_args()


def test_ResidualPlanarNavigateEnv_step():
    if args.auto_start_simulation:
        close_simulation()

    ENV = ResidualPlanarNavigateEnv  # PlanarNavigateEnv, ResidualPlanarNavigateEnv, YawControlEnv
    env_kwargs = {
        "DBG": True,
        "simulation": {
            "gui": True,
            "enable_meshes": True,
            "auto_start_simulation": args.auto_start_simulation,
            "enable_wind": False,
            "enable_wind_sampling": False,
            "enable_buoyancy_sampling": False,
            "wind_speed": 0,
            "wind_direction": (1, 0),
            "position": (0, 0, 100),  # initial spawned position
        },
        "observation": {
            "DBG_ROS": False,
            "DBG_OBS": False,
            "noise_stdv": 0.02,
        },
        "action": {
            "DBG_ACT": True,
            "act_noise_stdv": 0.05,
            "disable_servo": True,
        },
        "target": {
            "type": "MultiGoal",
            "target_name_space": "goal_",
            # "new_target_every_ts": 1200,
            "DBG_ROS": False,
            "enable_random_goal": False,
            "trigger_dist": 5,
            "wp_list": [
                (40, 40, -100, 3),
                (40, -40, -100, 3),
                (-40, -40, -100, 3),
                (-40, 40, -100, 3),
            ],
        },
        "mixer_type": "absolute",
        "mixer_param": (0.5, 1),    # only use rl control
    }

    episode = 3

    env = ENV(copy.deepcopy(env_kwargs))
    env.reset()
    for e in range(episode):
        # for i in range(1200):
        while True:
            action = env.action_space.sample()
            action = np.zeros_like(action)  # [yaw, pitch, servo, thrust]
            action[0] = 0 # yaw
            action[1] = -0.9   # pitch, left/rightfin_joint_position_controller
            action[2] = 0   # servo
            action[3] = 0 # thrust
            obs, reward, terminal, info = env.step(action)
            # print(info)
            print(action)

            assert isinstance(reward, float)
            assert isinstance(terminal, bool)
            assert isinstance(info, dict)

    GazeboConnection().unpause_sim()



if __name__ == "__main__":
    test_ResidualPlanarNavigateEnv_step()