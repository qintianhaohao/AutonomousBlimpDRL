import copy
import numpy as np
from blimp_env.envs.common.gazebo_connection import GazeboConnection
from blimp_env.envs.script import close_simulation
from blimp_env.envs import ResidualPlanarNavigateEnv



if __name__ == "__main__":
    auto_start_simulation = False
    if auto_start_simulation:
        close_simulation()

    ENV = ResidualPlanarNavigateEnv  # PlanarNavigateEnv, ResidualPlanarNavigateEnv, YawControlEnv
    env_kwargs = {
        "DBG": True,
        "simulation": {
            "gui": True,
            "enable_meshes": True,
            "auto_start_simulation": auto_start_simulation,
            "enable_wind": False,
            "enable_wind_sampling": False,
            "enable_buoyancy_sampling": False,
            "wind_speed": 0,
            "wind_direction": (1, 0),
            "position": (0, 0, -30),  # initial spawned position
        },
        "observation": {
            "DBG_ROS": False,
            "DBG_OBS": False,
            "noise_stdv": 0.02,
        },
        "action": {
            "DBG_ACT": False,
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
        "mixer_param": (0.5, 0),    # only use pid control
    }

    def env_step():
        env = ENV(copy.deepcopy(env_kwargs))
        env.reset()

        for _ in range(100000):
            # action = env.action_space.sample()
            # action = np.zeros_like(action)  # [yaw, pitch, servo, thrust]
            action = env.base_ctrl()
            obs, reward, terminal, info = env.step(action)
            # print(info)
            print(action)

        GazeboConnection().unpause_sim()

    env_step()
