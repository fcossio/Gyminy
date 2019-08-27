from gym.envs.registration import register
#from gym.scoreboard.registration import add_task, add_group

import os
import os.path as osp
import subprocess

os.environ['QT_PLUGIN_PATH'] = osp.join(osp.dirname(osp.abspath(__file__)), '.qt_plugins') + ':' + \
                               os.environ.get('QT_PLUGIN_PATH','')

# Nao

register(
    id='RoboschoolNaoForwardWalk-v1',
    entry_point='roboschoolfork_nao.envs.gym_nao:RoboschoolNaoForwardWalk',
    max_episode_steps=1000,
    tags={ "pg_complexity": 200*1000000 },
    )
register(
    id='NaoLLC-v1',
    entry_point='roboschoolfork_nao.envs.gym_nao_LLC:NaoLLC',
    max_episode_steps=1000,
    tags={ "pg_complexity": 200*1000000 },
    )
from .envs.gym_nao import RoboschoolNaoForwardWalk
from .envs.gym_nao_LLC import NaoLLC
