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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO

from .go2.go2_config import Go2Cfg, Go2CfgPPO
from .go2.go2 import Go2

from .go2.go2_him_config import Go2HimRoughCfg, Go2HimRoughCfgPPO
from .go2.go2_him import Go2HimRough

from .reddog.reddog_him_config import ReddogHimRoughCfg, ReddogHimRoughCfgPPO
from .reddog.reddog_him import ReddogHimRough

from .reddog.reddog_config import ReddogCfg, ReddogCfgPPO
from .reddog.reddog import Reddog

from .anymal_c.anymal_him import AnymalCHimRough
from .anymal_c.mixed_terrains.anymal_c_him_config import AnymalCHimRoughCfg, AnymalCHimRoughCfgPPO

from .reddog_big.big_reddog_him_config import BigReddogHimRoughCfg, BigReddogHimRoughCfgPPO
from .reddog_big.big_reddog_him import BigReddogHimRough

from .reddog_big.big_reddog_rough_config import BigReddogRoughCfg, BigReddogRoughCfgPPO
from .reddog_big.big_reddog import BigReddog

from .bipedal.bipedal_him_config import BipedalHimRoughCfg, BipedalHimRoughCfgPPO
from .bipedal.bipedal_him import BipedalHimRough

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "anymal_c_him", AnymalCHimRough, AnymalCHimRoughCfg(), AnymalCHimRoughCfgPPO() )

task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "go2", Go2, Go2Cfg(), Go2CfgPPO())
task_registry.register( "go2_him", Go2HimRough, Go2HimRoughCfg(), Go2HimRoughCfgPPO() ) # note that 'him' must be included in the name of the task

task_registry.register( "reddog_him", ReddogHimRough, ReddogHimRoughCfg(), ReddogHimRoughCfgPPO() )
task_registry.register( "reddog", Reddog, ReddogCfg(), ReddogCfgPPO() )

task_registry.register( "big_reddog_him", BigReddogHimRough, BigReddogHimRoughCfg(), BigReddogHimRoughCfgPPO() )
task_registry.register( "big_reddog", BigReddog, BigReddogRoughCfg(), BigReddogRoughCfgPPO() )

task_registry.register( "bipedal_him", BipedalHimRough, BipedalHimRoughCfg(), BipedalHimRoughCfgPPO() )