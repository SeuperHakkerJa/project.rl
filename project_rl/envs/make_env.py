from typing import Optional

import gym
from gym.wrappers import RescaleAction, RecordVideo

from project_rl.envs.wrappers import EpisodeMonitor

def register_gym_env(name, **kwargs):
    """A decorator to register gym environments.
    Args:
        name (str): a unique id to register in gym.
    """
    ## DANGEROUS
    if name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[name]

    def _register(cls):
        entry_point="{}:{}".format(cls.__module__, cls.__name__)
        gym.register(name, entry_point=entry_point, **kwargs)
        return cls

    return _register


def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             flatten: bool = True
             ) -> gym.Env:
    all_envs=gym.envs.registry.all()
    env_ids=[env_spec.id for env_spec in all_envs]

    # make sure the env is registered (use decorator)
    assert env_name in env_ids
    env=gym.make(env_name)

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    # normalize action
    env=RescaleAction(env, -1.0, 1.0)
    if save_folder is not None:
        env=RecordVideo(env, save_folder)

    if add_episode_monitor:
        env = EpisodeMonitor(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
