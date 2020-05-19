import ray
from gym.wrappers import TimeLimit
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from custom_envs.corridor_env import CorridorEnv
from custom_models.corridor_net import CorridorNet

if __name__ == '__main__':
    register_env('CorridorEnv', lambda env_config: TimeLimit(CorridorEnv(env_config['length']),
                                                             max_episode_steps=env_config['length']))
    ModelCatalog.register_custom_model('CorridorNet', CorridorNet)

    ray.init(local_mode=True)

    tune.run(
        'PPO',
        stop={'episode_reward_mean': 0.9},
        config={
            'env': 'CorridorEnv',
            'env_config': {
                'length': tune.grid_search([5, 10, 50]),
            },

            'model': {
                'custom_model': 'CorridorNet',
                'custom_options': {},
            },

            'use_pytorch': True,
            'num_gpus': 0,
            'num_workers': 2
        },
        verbose=1,
    )
