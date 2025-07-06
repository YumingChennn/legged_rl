from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym.envs.base.him_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR

from .bipedal_him_config import BipedalHimRoughCfg

class BipedalHimRough(LeggedRobot):
    cfg : BipedalHimRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)

    def _init_buffers(self):
        super()._init_buffers()
        self.actuator_net_input = torch.zeros(self.num_envs*self.num_actions, 6, device=self.device, requires_grad=False)
        self.joint_pos_err_last = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.joint_vel_err_last = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.joint_pos_err_last_last = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.joint_vel_err_last_last = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.joint_pos_err_last[env_ids] = 0.
        self.joint_pos_err_last_last[env_ids] = 0.
        self.joint_vel_err_last[env_ids] = 0.
        self.joint_vel_err_last_last[env_ids] = 0.
        if self.cfg.commands.num_commands == 5:
            self.extras["episode"]["max_height"] = self.command_ranges["height"][1] 

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        # set highest base height as the lowest height command at the begining
        if self.cfg.commands.num_commands == 5:
            self.command_ranges["height"][0] = self.cfg.commands.init_height
            self.command_ranges["height"][1] = self.cfg.commands.init_height

    def compute_observations(self):
        if self.cfg.commands.num_commands == 5:
            current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                        self.commands[:, 4:5] * self.obs_scales.height_measurements,
                                        self.base_ang_vel  * self.obs_scales.ang_vel,
                                        self.projected_gravity,
                                        (self.dof_pos[:, :2] - self.default_dof_pos[:, :2]) * self.obs_scales.dof_pos,
                                        (self.dof_pos[:, 3:5] - self.default_dof_pos[:, 3:5]) * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions,
                                        ),dim=-1)
        else:
            current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                        self.base_ang_vel  * self.obs_scales.ang_vel,
                                        self.projected_gravity,
                                        (self.dof_pos[:, :2] - self.default_dof_pos[:, :2]) * self.obs_scales.dof_pos,
                                        (self.dof_pos[:, 3:5] - self.default_dof_pos[:, 3:5]) * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions,
                                        ),dim=-1)
        # add noise if needed (configured for height command)
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:self.num_one_step_obs]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[self.num_one_step_obs:(self.num_one_step_obs+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)
 

    def compute_termination_observations(self, env_ids):
        if self.cfg.commands.num_commands == 5:
            current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                        self.commands[:, 4:5] * self.obs_scales.height_measurements,
                                        self.base_ang_vel  * self.obs_scales.ang_vel,
                                        self.projected_gravity,
                                        (self.dof_pos[:, :2] - self.default_dof_pos[:, :2]) * self.obs_scales.dof_pos,
                                        (self.dof_pos[:, 3:5] - self.default_dof_pos[:, 3:5]) * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions,
                                        ),dim=-1)
        else:
            current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                        self.base_ang_vel  * self.obs_scales.ang_vel,
                                        self.projected_gravity,
                                        (self.dof_pos[:, :2] - self.default_dof_pos[:, :2]) * self.obs_scales.dof_pos,
                                        (self.dof_pos[:, 3:5] - self.default_dof_pos[:, 3:5]) * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions,
                                        ),dim=-1)
        # add noise if needed (configured for height command)
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:self.num_one_step_obs]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[self.num_one_step_obs:(self.num_one_step_obs+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]

    def _get_noise_scale_vec(self, cfg):
        if self.cfg.terrain.measure_heights:
            noise_vec = torch.zeros((self.num_one_step_obs+187), device=self.device)
        else:
            noise_vec = torch.zeros(self.num_one_step_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        if self.cfg.commands.num_commands == 5:
            noise_vec[0:4] = 0. # commands
            noise_vec[4:7] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            noise_vec[7:10] = noise_scales.gravity * noise_level
            noise_vec[10:14] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[14:20] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[20:26] = 0. # previous actions
        else:
            noise_vec[0:3] = 0. # commands
            noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            noise_vec[6:9] = noise_scales.gravity * noise_level
            noise_vec[9:13] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[13:19] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[19:25] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[self.num_one_step_obs:(self.num_one_step_obs + 187)] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
            
        return noise_vec
            

    def _resample_commands(self, env_ids):
        if self.cfg.commands.num_commands == 5:
            self.commands[env_ids, 0] = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 4] = torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

            high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
            high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]

            self.commands[high_vel_env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(high_vel_env_ids), 1), device=self.device).squeeze(1)
        
            # set y commands of high vel envs to zero
            self.commands[high_vel_env_ids, 1:2] *= (torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1) < 1.0).unsqueeze(1)

            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        else:
            super()._resample_commands(env_ids)
    
    def update_command_curriculum(self, env_ids):
        if self.cfg.commands.num_commands == 5:
            low_vel_env_ids = (env_ids > (self.num_envs * 0.2))
            high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
            low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
            high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
            
            current_reward = torch.mean(self.episode_sums["base_height"]) / self.max_episode_length
            print('reward', current_reward)
            print('reward scale', self.reward_scales["base_height"])

            # If the tracking reward is above 80% of the maximum, increase the range of commands
            if (torch.mean(self.episode_sums["tracking_lin_vel"][low_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]) and (torch.mean(self.episode_sums["tracking_lin_vel"][high_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]):
                self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
                self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)

            if current_reward > 0.8 * 0.01:
                self.command_ranges["height"][0] = np.clip(self.command_ranges["height"][0] - 0.01, self.cfg.commands.ranges.height[0], self.cfg.commands.ranges.height[1])
                self.command_ranges["height"][1] = np.clip(self.command_ranges["height"][1] + 0.01, self.cfg.commands.ranges.height[0], self.cfg.commands.ranges.height[1])
        else:
            super().update_command_curriculum(env_ids)

    def _compute_torques(self, actions):
        if self.cfg.control.use_actuator_network:
            action_scaled = actions * self.cfg.control.action_scale
            joint_pos_des = self.default_dof_pos + action_scaled
            joint_pos_err = self.dof_pos - joint_pos_des    # shape: (num_envs, num_actions)
            joint_vel_err = self.dof_vel    # shape: (num_envs, num_actions)
            with torch.inference_mode():
                self.actuator_net_input = torch.cat((
                    joint_pos_err.unsqueeze(-1),
                    self.joint_pos_err_last.unsqueeze(-1),
                    self.joint_pos_err_last_last.unsqueeze(-1),
                    joint_vel_err.unsqueeze(-1),
                    self.joint_vel_err_last.unsqueeze(-1),
                    self.joint_vel_err_last_last.unsqueeze(-1)
                ), dim=2).view(-1, 6)   # shape: (num_envs*num_actions, 6)
                torques = self.actuator_network(self.actuator_net_input).view(self.num_envs, self.num_actions)
                
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_vel_err_last_last = torch.clone(self.joint_vel_err_last)
            self.joint_pos_err_last = torch.clone(joint_pos_err)
            self.joint_vel_err_last = torch.clone(joint_vel_err)
            
            return torques
            
        else:
            control_type = self.cfg.control.control_type

            if control_type == "MIXED_LIMBS_PV":
                # def _compute_torques(self, actions):
                """Compute torques using mixed position and velocity control."""
                # 控制參數與 index
                pos_indices = self.cfg.control.position_control_indices  # e.g., [0, 1, 3, 4]
                vel_indices = self.cfg.control.velocity_control_indices  # e.g., [2, 5]

                pos_action_scale = self.cfg.control.pos_action_scale     # e.g., 0.5
                vel_action_scale = self.cfg.control.vel_action_scale     # e.g., 10.0

                # 預設 torque = 0
                torques = torch.zeros_like(self.dof_pos)

                # === Position Control ===
                if len(pos_indices) > 0:
                    pos_idx = torch.tensor(pos_indices, device=actions.device)
                    pos_actions = actions[:, pos_idx] * pos_action_scale
                    target_pos = self.default_dof_pos[:, pos_idx] + pos_actions
                    pos_err = target_pos - self.dof_pos[:, pos_idx]
                    vel_err = self.dof_vel[:, pos_idx]

                    # Kp_factors: [4096, 1] 會自動 broadcast 為 [4096, len(pos_idx)]
                    torques[:, pos_idx] = (
                        self.p_gains[pos_idx] * self.Kp_factors * pos_err
                        - self.d_gains[pos_idx] * self.Kd_factors * vel_err
                    )

                # === Velocity Control ===
                if len(vel_indices) > 0:
                    vel_idx = torch.tensor(vel_indices, device=actions.device)
                    vel_actions = actions[:, vel_idx] * vel_action_scale
                    vel_err = vel_actions - self.dof_vel[:, vel_idx]
                    torques[:, vel_idx] = self.d_gains[vel_idx] * vel_err  # 不用 D term（可加）
                # print(torques[0,:])
                # Clip 最終 torque 輸出
                return torch.clip(torques, -self.torque_limits, self.torque_limits)
            else:
                raise NameError(f"Unknown controller type: {control_type}")

    # ------------ reward functions----------------
    # def _reward_tracking_lin_vel_enhance(self):
    #     # Tracking of linear velocity commands (x axes)
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma / 10) - 1
    
    def _reward_nominal_state_thigh(self):
        diff = self.dof_pos[:, 0] - self.dof_pos[:, 3]
        return torch.square(diff)
    
    def _reward_nominal_state_calf(self):
        diff = self.dof_pos[:, 1] - self.dof_pos[:, 4]
        return torch.square(diff)
    
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, :2]), dim=1) + torch.sum(
            torch.square(self.dof_vel[:, 3:5]), dim=1
        )
    
    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions[:, :2]
                - 2 * self.last_actions[:, :2]
                + self.last_last_actions[:, :2]
            ),
            dim=1,
        ) + torch.sum(
            torch.square(
                self.actions[:, 3:5]
                - 2 * self.last_actions[:, 3:5]
                + self.last_last_actions[:, 3:5]
            ),
            dim=1,
        )

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos[:, :2] - self.dof_pos_limits[:2, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos[:, :2] - self.dof_pos_limits[:2, 1]).clip(
            min=0.0
        )
        out_of_limits += -(self.dof_pos[:, 3:5] - self.dof_pos_limits[3:5, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos[:, 3:5] - self.dof_pos_limits[3:5, 1]).clip(
            min=0.0
        )
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_base_height(self):
        base_height = self._get_base_heights()
        if self.cfg.commands.num_commands == 5:
            if self.reward_scales["base_height"] < 0:
                return torch.square(base_height - self.commands[:, 4])
            else:
                base_height_error = torch.square(base_height - self.commands[:, 4])
                return torch.exp(-base_height_error / 0.001)
        else:
            if self.reward_scales["base_height"] < 0:
                return torch.square(base_height - self.cfg.rewards.base_height_target)
            else:
                base_height_error = torch.square(base_height - self.cfg.rewards.base_height_target)
                return torch.exp(-base_height_error / 0.001)