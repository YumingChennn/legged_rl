This repo refer to [zitongbai/legged_rl](https://github.com/zitongbai/legged_rl.git)

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
   - For troubleshooting check docs `isaacgym/docs/index.html`
4. Install rsl_rl (PPO implementation)
   - remove existed rsl_rl folder
   - Go to [rsl_rl](https://github.com/zitongbai/rsl_rl.git) and clone the repo
   - make sure you have activated the conda env
   - In rsl_rl: `git checkout v1.0.2 && pip install -e .` 
   - check back to 6d5b057: `git checkout "6d5b057"`
5. Install legged_gym
   - We have a forked version in this repo.
   - make sure you have activated the conda env
   - In this repo: `cd legged_gym && pip install -e .`

You can refer to https://github.com/leggedrobotics/legged_gym for detailed information. 

# Usage

## Train and play
### reddog
Train big reddog with PPO algorithm: 
```bash
# in legged_rl\legged_gym
python legged_gym/scripts/train.py --task=big_reddog --headless --max_iterations=1000
```

After training, play once to export the jit file:
```bash
# in legged_rl\legged_gym
python legged_gym/scripts/play.py --task=big_reddog
```

### wheel-biped robot
Train wheel-biped robot with HimLoco algorithm: 
```bash
# in legged_rl\legged_gym
python legged_gym/scripts/train.py --task=bipedal_him --headless --max_iterations=1500
```

After training, play once to export the jit file:
```bash
# in legged_rl\legged_gym
python legged_gym/scripts/play_him_bipedal.py --task=bipedal_him
```
