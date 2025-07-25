# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.him_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class ReddogHimRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.34] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': -0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': 0.1,   # [rad]

            'FL_thigh_joint': 0.785,     # [rad]
            'RL_thigh_joint': -0.785,   # [rad]
            'FR_thigh_joint': 0.785,     # [rad]
            'RR_thigh_joint': -0.785,   # [rad]

            'FL_calf_joint': -1.57,   # [rad]
            'RL_calf_joint': 1.57,    # [rad]
            'FR_calf_joint': -1.57,  # [rad]
            'RR_calf_joint': 1.57,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 10}  # [N*m/rad] 3
        damping = {'joint': 0.4}     # [N*m*s/rad] 0.1
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25 #  0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0
        use_actuator_network = False
        # actuator_net_file = '{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/go2_actuator_net.pt'

    # class commands( LeggedRobotCfg.commands ):
    #         curriculum = True # True
    #         max_curriculum = 1.0 # 2.0
    #         num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
    #         resampling_time = 5. # 10. # time before command are changed[s]
    #         heading_command = True # if true: compute ang vel command from heading error
    #         class ranges( LeggedRobotCfg.commands.ranges):
    #             lin_vel_x = [-0.5, 0.5] # min max [m/s]
    #             lin_vel_y = [-0.5, 0.5]   # min max [m/s]
    #             ang_vel_yaw = [-0.8, 0.8]    # min max [rad/s]
    #             heading = [-3.14, 3.14]

    # class terrain( LeggedRobotCfg.terrain ):
    #     mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
    #     vertical_scale = 0.002 # m

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/reddog/urdf/reddog.urdf'
        name = "reddog"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up

    # class domain_rand(LeggedRobotCfg.domain_rand):
        # randomize_payload_mass = True 
        # payload_mass_range = [-0.3, 0.3]

        # randomize_com_displacemreward_scalesent = True
        # com_displacement_range = [-0.03, 0.03] # [-0.05, 0.05] 

        # randomize_link_mass = False # True
        # link_mass_range = [0.9, 1.1]
        
        # randomize_friction = True
        # friction_range = [0.2, 1.25] # [0.2, 1.25]
        
        # randomize_restitution = False
        # restitution_range = [0., 1.0]
        
        # randomize_motor_strength = True
        # motor_strength_range = [0.1, 0.3]
        
        # randomize_kp = True
        # kp_range = [0.9, 1.1]
        
        # randomize_kd = False
        # kd_range = [0.9, 1.1]
        
        # randomize_initial_joint_pos = True
        # initial_joint_pos_range = [0.5, 1.5]
        
        # disturbance = True
        # disturbance_range = [-30.0, 30.0]
        # disturbance_interval = 8
        
        # push_robots = True
        # push_interval_s = 16
        # max_push_vel_xy = 1.

        # delay = True

  
    # class rewards( LeggedRobotCfg.rewards ):
    #     class scales:
    #         termination = -0.0
    #         tracking_lin_vel = 1.0
    #         tracking_ang_vel = 0.5
    #         lin_vel_z = -2.0
    #         ang_vel_xy = -0.05
    #         orientation = -0.2
    #         dof_acc = 0.0
    #         joint_power = -2e-5
    #         base_height = -1.0
    #         foot_clearance = -0.01
    #         action_rate = -0.01
    #         smoothness = -0.01
    #         feet_air_time =  0.8 # 0.0
    #         collision = -0.0
    #         feet_stumble = -0.0
    #         stand_still = -0.
    #         torques = -0.0
    #         dof_vel = -0.0
    #         dof_pos_limits = -0.0
    #         dof_vel_limits = -0.0
    #         torque_limits = -0.0

    #     only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
    #     tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    #     soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    #     soft_dof_vel_limit = 1.
    #     soft_torque_limit = 1.
    #     base_height_target = 0.12
    #     max_contact_force = 100. # forces above this value are penalized
    #     clearance_height_target = -0.20

class ReddogHimRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_Reddog'
        
        save_interval = 100 # check for potential saves every this many iterations
        max_iterations = 1500 # number of policy updates

  