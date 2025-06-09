# Installation

This repo is tested on **Ubuntu 22.04** with ROS Humble.

## Legged gym

Now you have built the deployment part of this repo. The following is for training the RL policy.

1. Create a conda env with python 3.6, 3.7 or 3.8 (3.8 recommended) to train and play RL policy.
   - `conda create -n legged_rl python=3.8`
   - `conda activate legged_rl`
   - Please note that this conda env is only for training RL policy. It is **NOT** used in the deployment part. More precisely, **neither** the ROS workspace **nor** running the RL policy uses this conda env. 
2. Install [pytorch](https://pytorch.org/) in the conda env.
   - `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
3. Install Isaac Gym (not necessary in this workspace)
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - make sure you have activated the conda env
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   - We have a forked version in this repo.
   - make sure you have activated the conda env
   - In this repo: `cd rsl_rl && git checkout v1.0.2 && pip install -e .` 
5. Install legged_gym
   - We have a forked version in this repo.
   - make sure you have activated the conda env
   - In this repo: `cd legged_gym && pip install -e .`

You can refer to https://github.com/leggedrobotics/legged_gym for detailed information. 

# Usage

## Train and export
Train go2 with [HIMLoco](https://github.com/OpenRobotLab/HIMLoco/blob/main/projects/himloco/README.md) algorithm: 
```bash
# in legged_rl\legged_gym
python legged_gym/scripts/train.py --task=go2_him --headless --max_iterations=1000
```

After training, play once to export the jit file:
```bash
# in legged_rl\legged_gym
python legged_gym/scripts/play_him.py --task=go2_him
```

Export rl config: 
```bash
# in legged_rl
python legged_gym/legged_gym/scripts/export_config.py --target_dir=legged_robot_example/unitree_go2_description/config --task=go2_him
```
