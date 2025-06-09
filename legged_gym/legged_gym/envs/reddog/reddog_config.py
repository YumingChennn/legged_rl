from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class ReddogCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #     'FL_hip_joint': 0.1,   # [rad]
        #     'RL_hip_joint': -0.1,   # [rad]
        #     'FR_hip_joint': -0.1 ,  # [rad]
        #     'RR_hip_joint': 0.1,   # [rad]

        #     'FL_thigh_joint': 0.4,     # [rad] #0.785
        #     'RL_thigh_joint': -0.4,   # [rad]
        #     'FR_thigh_joint': 0.4,     # [rad]
        #     'RR_thigh_joint': -0.4,   # [rad]

        #     'FL_calf_joint': -0.8,   # [rad] #0.8
        #     'RL_calf_joint': 0.8,    # [rad]
        #     'FR_calf_joint': -0.8,  # [rad]
        #     'RR_calf_joint': 0.8,    # [rad]
        # }
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.0,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.0,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 15.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class commands ( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 0.8 # 2.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.25, 0.25] # min max [m/s]
            lin_vel_y = [-0.25, 0.25]   # min max [m/s]
            ang_vel_yaw = [-0.785, 0.785]    # min max [rad/s]
            heading = [-0.785, 0.785]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/reddogv2/urdf/reddogv2.urdf'
        name = "reddog"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        # privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        added_mass_range = [-0.3, 0.3]
        max_push_vel_xy = 0.5
        # randomize_payload_mass = True 
        # payload_mass_range = [-0.3, 0.3]

        # randomize_com_displacement = True
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
    #     soft_dof_pos_limit = 0.9
    #     base_height_target = 0.33
    #     # max_contact_force = 200
    #     class scales( LeggedRobotCfg.rewards.scales ):
    #         # orientation = -5.0
    #         # feet_air_time = 2.
    #         torques = -0.0002
    #         dof_pos_limits = -10.0
    #         # base_height = -0.1
            
    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            torques = -0.00001 #-0.000025
            dof_vel = -0
            dof_acc = -2.5e-7 #-2.5e-7
            # base_height = -0.0
            feet_air_time =  0.2
            collision = -0.001
            # feet_stumble = -0.01
            action_rate = -0.01
            stand_still = -0.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.22 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = 0.07

class ReddogCfgPPO( LeggedRobotCfgPPO ):
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'reddog'
        max_iterations = 500 # number of policy updates

  