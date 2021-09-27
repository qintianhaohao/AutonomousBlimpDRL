# Autonomous Blimp Control using Deep Reinforcement Learning
=================================================================

## For more information, read our preprint on arXiv: https://arxiv.org/abs/2109.10719
--------------------------------------------------------------

# Copyright and License

All Code in this repository - unless otherwise stated in local license or code headers is

Copyright 2021 Max Planck Institute for Intelligent Systems

Licensed under the terms of the GNU General Public Licence (GPL) v3 or higher.
See: https://www.gnu.org/licenses/gpl-3.0.en.html


# Contents

* /RL -- RL agent related files.
* /blimp_env -- training environment of the RL agent. 
* /path_planner -- waypoints assignment.

# Install blimp simulator
See: https://github.com/robot-perception-group/airship_simulation. 

Note: The default ROS version for the blimp simulator is melodic, you can switch to noetic branch.

## Configure software-in-the-loop firmware
This step has 2 purposes:
*  enable ROS control on the firmware
*  start firmware without Librepilot GUI

1. In the firts terminal starts the firmware
```
cd ~/catkin_ws/src/airship_simulation/LibrePilot
./build/firmware/fw_simposix/fw_simposix.elf 0  
```

2. Start the gcs in the second terminal
```
cd ~/catkin_ws/src/airship_simulation/LibrePilot
./build/librepilot-gcs_release/bin/librepilot-gcs
```
3. Select "Connections" (bottom right) --> UDP: localhost --> Click "Connect"
4. "Configuration" tab --> "Input" tab (left) --> "Arming Setting" --> Change "Always Armed" to "Always Disarmed" --> Click "Apply"
5. "HITL" tab --> click "Start" --> check "GCS Control" 
   This will disarm the firmware and allow to save the configuration
6. "Configuration" tab --> "Input" tab (left) --> "Flight Mode Switch Settings" --> Change Flight Mode Pos. 1 from "Manual" to "ROSControlled" 
7. "Configuration" tab --> "Input" tab (left) --> "Arming Setting" --> Change "Always Disarmed" to "Always Armed" --> Click "Save" --> Click "Apply" 
8. Confirm the change by restart firmware, connect via gcs, and check if Flight Mode Pos.1 is changed to "ROSControlled"


# Install RL training environment
```console
git clone https://github.com/robot-perception-group/AutonomousBlimpDRL.git
```

preparing...

# Cite
```
@article{Liu2021ABCDRL,
  title={Autonomous Blimp Control using Deep Reinforcement Learning},
  author={Yu Tang Liu, Eric Price, Pascal Goldschmid, Michael J. Black, Aamir Ahmad},
  journal={arXiv preprint arXiv:2109.10719},
  year={2021}
}
```
