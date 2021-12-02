""" navigate env with velocity target """
#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rospy
from blimp_env.envs.common.abstract import ROSAbstractEnv
from blimp_env.envs.common.action import Action
from geometry_msgs.msg import Point, Quaternion
from blimp_env.envs.script.blimp_script import respawn_model
import line_profiler

profile = line_profiler.LineProfiler()

Observation = Union[np.ndarray, float]


class PlanarNavigateEnv(ROSAbstractEnv):
    """Navigate blimp by path following decomposed to altitude and planar control"""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config["observation"].update(
            {
                "type": "PlanarKinematics",
                "noise_stdv": 0.02,
                "scale_obs": True,
            }
        )
        config["action"].update(
            {
                "type": "SimpleContinuousDifferentialAction",
                "act_noise_stdv": 0.05,
                "disable_servo": True,
            }
        )
        config["target"].update(
            {
                "type": "RandomGoal",
                "target_name_space": "goal_",
                "new_target_every_ts": 1200,
            }
        )
        config.update(
            {
                "duration": 1200,
                "simulation_frequency": 30,  # [hz]
                "policy_frequency": 6,  # [hz] has to be greater than 5 to overwrite backup controller
                "reward_weights": np.array([1, 0.7, 0.2]),  # success, tracking, action
                "tracking_reward_weights": np.array(
                    [0.25, 0.25, 0.25, 0.25]
                ),  # z_diff, planar_dist, psi_diff, vel_diff
                "success_threshhold": 5,  # [meters]
            }
        )
        return config

    def _create_pubs_subs(self):
        super()._create_pubs_subs()
        self.rew_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_reward", Quaternion, queue_size=1
        )
        self.state_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_state", Quaternion, queue_size=1
        )
        self.vel_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_vel", Point, queue_size=1
        )
        self.vel_diff_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_vel_diff", Point, queue_size=1
        )
        self.ang_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_ang", Point, queue_size=1
        )
        self.ang_diff_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_ang_diff", Point, queue_size=1
        )
        self.act_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_act", Quaternion, queue_size=1
        )
        self.pos_cmd_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_pos_cmd",
            Point,
            queue_size=1,
        )

    @profile
    def one_step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """[perform a step action and observe result]

        Args:
            action (Action): action from the agent [-1,1] with size (4,)

        Returns:
            Tuple[Observation, float, bool, dict]:
                obs: np.array [-1,1] with size (9,),
                reward: scalar,
                terminal: bool,
                info: dictionary of all the step info,
        """
        self._simulate(action)
        obs, obs_info = self.observation_type.observe()
        reward, reward_info = self._reward(obs, action, obs_info)
        terminal = self._is_terminal(obs_info)
        info = {
            "step": self.steps,
            "obs": obs,
            "obs_info": obs_info,
            "act": action,
            "reward": reward,
            "reward_info": reward_info,
            "terminal": terminal,
        }

        self._update_goal()
        self._step_info(info)

        return obs, reward, terminal, info

    def _step_info(self, info: dict):
        """publish all the step information to rviz

        Args:
            info ([dict]): [dict contain all step information]
        """
        obs_info = info["obs_info"]
        proc_info = obs_info["proc_dict"]

        self.rew_rviz_pub.publish(Quaternion(*info["reward_info"]["rew_info"]))
        self.state_rviz_pub.publish(
            Quaternion(
                proc_info["planar_dist"],
                proc_info["psi_diff"],
                proc_info["z_diff"],
                proc_info["vel_diff"],
            )
        )
        self.vel_rviz_pub.publish(Point(*obs_info["velocity"]))
        self.vel_diff_rviz_pub.publish(
            Point(
                obs_info["velocity_norm"],
                self.goal["velocity"],
                proc_info["vel_diff"],
            )
        )
        self.ang_rviz_pub.publish(Point(*obs_info["angle"]))
        self.ang_diff_rviz_pub.publish(Point(0, 0, proc_info["psi_diff"]))
        self.act_rviz_pub.publish(Quaternion(*info["act"]))

        self.pos_cmd_pub.publish(Point(*self.goal["position"]))

        if self.dbg:
            print(
                f"================= [ PlanarNavigateEnv ] step {self.steps} ================="
            )
            print("STEP INFO:", info)
            print("\r")

    def reset(self) -> Observation:
        self.steps = 0
        self.done = False
        self._reset()
        obs, _ = self.observation_type.observe()
        return obs

    def _update_goal(self):
        """sample new goal dictionary"""
        self.goal = self.target_type.sample()

    def _reward(
        self, obs: np.array, act: np.array, obs_info: dict
    ) -> Tuple[float, dict]:
        """calculate reward
        total_reward = success_reward + tracking_reward + action_reward
        success_reward: +1 if agent stay in the vicinity of goal
        tracking_reward: - L2 distance to goal - psi angle difference - z diff - vel diff
        action_reward: penalty for motor use

        Args:
            obs (np.array): ("z_diff", "planar_dist", "psi_diff", "vel_diff", "vel", "psi_vel" "action")
            act (np.array): agent action [-1,1] with size (4,)
            obs_info (dict): contain all information of a step

        Returns:
            Tuple[float, dict]: [reward scalar and a detailed reward info]
        """
        track_weights = self.config["tracking_reward_weights"].copy()
        reward_weights = self.config["reward_weights"].copy()

        success_reward = self.compute_success_rew(
            obs_info["position"], self.goal["position"]
        )
        obs[1] = (obs[1] + 1) / 2  # dist -1 should have max reward
        tracking_reward = np.dot(track_weights, -np.abs(obs[0:4]))
        action_reward = self.action_type.action_rew()

        reward = np.dot(
            reward_weights,
            (success_reward, tracking_reward, action_reward, ),
        )
        reward = np.clip(reward, -1, 1)
        rew_info = (reward, success_reward, tracking_reward, action_reward)
        reward_info = {"rew_info": rew_info}

        return reward, reward_info

    def compute_success_rew(self, pos: np.array, goal_pos: np.array) -> float:
        """task success if distance to goal is less than sucess_threshhold

        Args:
            pos ([np.array]): [position of machine]
            goal_pos ([np.array]): [position of planar goal]
            k (float): scaler for success

        Returns:
            [float]: [1 if success, otherwise 0]
        """
        return (
            1.0
            if np.linalg.norm(pos[0:2] - goal_pos[0:2]) < self.config["success_threshhold"]
            else 0.0
        )

    def _is_terminal(self, obs_info: dict) -> bool:
        """if episode terminate
        - time: episode duration finished

        Returns:
            bool: [episode terminal or not]
        """
        time = False
        if self.config["duration"] is not None:
            time = self.steps >= int(self.config["duration"])-1

        success_reward = self.compute_success_rew(
            obs_info["position"], self.goal["position"]
        )
        success = success_reward >= 1

        return time or success

    def close(self) -> None:
        return super().close()


class ResidualPlanarNavigateEnv(PlanarNavigateEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config["observation"].update(
            {
                "type": "PlanarKinematics",
                "noise_stdv": 0.015,
                "scale_obs": True,
            }
        )
        config["action"].update(
            {
                "type": "SimpleContinuousDifferentialAction",
                "act_noise_stdv": 0.05,
                "disable_servo": False,
            }
        )
        config["target"].update(
            {
                "type": "RandomGoal",
                "target_name_space": "goal_",
                "new_target_every_ts": 1200,
            }
        )
        config.update(
            {
                "duration": 1200,
                "simulation_frequency": 30,  # [hz]
                "policy_frequency": 10,  # [hz] has to be greater than 5 to overwrite backup controller
                "reward_weights": np.array([20, 0.8, 0.1, 0.1]),  # success, tracking, action, bonus
                "tracking_reward_weights": np.array(
                    [0.25, 0.35, 0.25, 0.15]
                ),  # z_diff, planar_dist, psi_diff, vel_diff
                "success_threshhold": 5,  # [meters]
            }
        )
        return config

    def __init__(self, config: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(config=config)

        self.delta_t=0.01
        self.psi_err_i, self.alt_err_i, self.vel_err_i = 0, 0, 0
        self.prev_psi, self.prev_alt, self.prev_vel = 0, 0, 0

    def _create_pubs_subs(self):
        self.rew_bonus_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_reward_psi_bonus", Point, queue_size=1
        )
        self.ang_vel_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_ang_vel", Point, queue_size=1
        )
        self.residual_act_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_residual_act", Quaternion, queue_size=1
        )
        return super()._create_pubs_subs()

    @profile
    def one_step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """[perform a step action and observe result]

        Args:
            action (Action): action from the agent [-1,1] with size (4,)

        Returns:
            Tuple[Observation, float, bool, dict]:
                obs: np.array [-1,1] with size (9,),
                reward: scalar,
                terminal: bool,
                info: dictionary of all the step info,
        """
        residual_act = self.residual_ctrl()
        joint_act = self.mixer(action, residual_act)
        self._simulate(joint_act)
        obs, obs_info = self.observation_type.observe()
        reward, reward_info = self._reward(obs, joint_act, obs_info)
        terminal = self._is_terminal(obs_info)
        info = {
            "step": self.steps,
            "obs": obs,
            "obs_info": obs_info,
            "act": action,
            "residual_act": residual_act,
            "joint_act": joint_act,
            "reward": reward,
            "reward_info": reward_info,
            "terminal": terminal,
        }

        self._update_goal()
        self._step_info(info)

        return obs, reward, terminal, info
    
    def _step_info(self, info: dict):
        obs_info = info["obs_info"]
        proc_info = obs_info["proc_dict"]

        self.rew_bonus_rviz_pub.publish(Point(*info["reward_info"]["bonus_info"]))
        self.ang_vel_rviz_pub.publish(Point(0, 0, proc_info["psi_vel"]))
        self.residual_act_rviz_pub.publish(Quaternion(*info["residual_act"]))
        return super()._step_info(info)
    
    def mixer(self, action, residual_act):
        joint_act = 0.5*action+0.5*residual_act
        joint_act[2] = action[2]
        return joint_act
    
    def residual_ctrl(self):
        """      
        use PID controller to generate a residual signal
        """
        obs, _ = self.observation_type.observe()

        psi_ctrl, self.psi_err_i, self.prev_psi = self.pid_ctrl(-obs[2], self.psi_err_i, self.prev_psi, np.array([1,.0,.5]))
        alt_ctrl, self.alt_err_i, self.prev_alt = self.pid_ctrl(obs[0], self.alt_err_i, self.prev_alt)
        vel_ctrl, self.vel_err_i, self.prev_vel = self.pid_ctrl(-obs[3], self.vel_err_i, self.prev_vel)
        ser_ctrl = 0
        
        residual_act = np.array([psi_ctrl, alt_ctrl, ser_ctrl, vel_ctrl])
        residual_act = np.clip(residual_act, -1, 1) 
        return residual_act
    
    def clear_integrater(self):
        self.psi_err_i, self.alt_err_i, self.vel_err_i = 0, 0, 0
        self.prev_psi, self.prev_alt, self.prev_vel = 0, 0, 0
    
    def pid_ctrl(self, err, err_i, err_prev, pid_coeff=np.array([1, 0.5, 0.3])):
        err_i += err*self.delta_t
        err_i = np.clip(err_i, -1, 1)
        err_d = (err-err_prev)/self.delta_t
        ctrl = np.dot(pid_coeff, np.array([err, err_i, err_d]))
        return ctrl, err_i, err

    def reset(self) -> Observation:
        self.clear_integrater()
        return super().reset()

    def _reward(
        self, obs: np.array, act: np.array, obs_info: dict
    ) -> Tuple[float, dict]:
        """calculate reward
        total_reward = success_reward + tracking_reward + action_reward
        success_reward: +1 if agent stay in the vicinity of goal
        tracking_reward: - L2 distance to goal - psi angle difference - z diff - vel diff
        action_reward: penalty for motor use

        Args:
            obs (np.array): ("z_diff", "planar_dist", "psi_diff", "vel_diff", "vel", "psi_vel" "action")
            act (np.array): agent action [-1,1] with size (4,)
            obs_info (dict): contain all information of a step

        Returns:
            Tuple[float, dict]: [reward scalar and a detailed reward info]
        """
        track_weights = self.config["tracking_reward_weights"].copy()
        reward_weights = self.config["reward_weights"].copy()

        success_reward = self.compute_success_rew(
            obs_info["position"], self.goal["position"]
        )

        obs[1] = (obs[1] + 1) / 2  # dist -1 should have max reward
        tracking_reward = np.dot(track_weights, -np.abs(obs[0:4]))
        
        action_reward = self.action_type.action_rew()

        psi_diff, psi_vel, psi_acc = obs[2], obs[5], -obs[6]
        psi_sign_bonus = 0.5*np.abs(psi_diff)*(np.sign(psi_diff)*(0.7*np.sign(psi_vel)+0.3*np.sign(psi_acc))-1)
        psi_close_bonus = -5*int(self.psi_close_to_pi())
        bonus = psi_sign_bonus+psi_close_bonus

        reward = np.dot(
            reward_weights,
            (success_reward, tracking_reward, action_reward, bonus),
        )
        # reward = np.clip(reward, -1, 1)
        rew_info = (reward, success_reward, tracking_reward, action_reward)
        bonus_info = (bonus, psi_sign_bonus, psi_close_bonus)
        reward_info = {"rew_info": rew_info, "bonus_info": bonus_info}

        return reward, reward_info

    def _is_terminal(self, obs_info: dict) -> bool:
        """if episode terminate
        - time: episode duration finished

        Returns:
            bool: [episode terminal or not]
        """
        time = False
        if self.config["duration"] is not None:
            time = self.steps >= int(self.config["duration"])-1

        success_reward = self.compute_success_rew(
            obs_info["position"], self.goal["position"]
        )
        success = success_reward >= 1

        early_stopping = bool(self.psi_change())

        return time or success or early_stopping

    def psi_change(self):
        """[detect if psi changes from -pi to pi and vice versa]

        Returns:
            [bool]: [if psi changes from -pi to pi and vice versa]
        """
        obs, _ = self.observation_type.observe()
        return np.abs(-obs[2]-self.prev_psi)>1.9 

    def psi_close_to_pi(self):
        """[detect if psi is close to pi and -pi]

        Returns:
            [bool]: []
        """
        obs, _ = self.observation_type.observe()
        return np.abs(np.abs(obs[2])-1)<0.05


        

if __name__ == "__main__":
    import copy
    from blimp_env.envs.common.gazebo_connection import GazeboConnection
    from blimp_env.envs.script import close_simulation

    # ============== profile ==============#
    # 1. pip install line_profiler
    # 2. in terminal:
    # kernprof -l -v blimp_env/envs/planar_navigate_env.py

    auto_start_simulation = False
    if auto_start_simulation:
        close_simulation()

    # ENV = PlanarNavigateEnv
    ENV = ResidualPlanarNavigateEnv
    env_kwargs = {
        "DBG": True,
        "simulation": {
            "gui": True,
            "enable_meshes": True,
            "auto_start_simulation": auto_start_simulation,
        },
        "observation": {
            "DBG_ROS": False,
            "DBG_OBS": False,
            "noise_stdv": 0.0,
        },
        "action": {
            "DBG_ACT": False,
            "act_noise_stdv": 0.0,
        },
        "target": {"DBG_ROS": False},
    }

    @profile
    def env_step():
        env = ENV(copy.deepcopy(env_kwargs))
        env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            action = np.array([0.0, 0, 0, 0]) # [yaw, pitch, servo, thrust]
            obs, reward, terminal, info = env.step(action)

        GazeboConnection().unpause_sim()

    env_step()


