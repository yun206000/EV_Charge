import stable_baselines3

"""
The project runs in two environments (README_for_doc.md section 11.10):

    PCRL        Python 3.8, gym 0.21, stable-baselines3 1.5 (the trained models and all reported results)
    PCRL-py312  Python 3.12, gymnasium, stable-baselines3 2.x (current packages without known vulnerabilities)

stable-baselines3 2.x uses gymnasium, whose reset() returns (obs, info) and step() returns
(obs, reward, terminated, truncated, info) instead of obs and (obs, reward, done, info). The environments return the
form of the installed library; the scripts call reset() / step() below and always get the old form.
"""

GYMNASIUM = int(stable_baselines3.__version__.split(".")[0]) >= 2

if GYMNASIUM:
    import gymnasium as gym
else:
    import gym

spaces = gym.spaces


def reset_result(obs):
    """return value of Env.reset() for the installed library"""
    return (obs, {}) if GYMNASIUM else obs


def step_result(obs, reward, done, info):
    """return value of Env.step() for the installed library (an episode only ends by its own rules: terminated)"""
    return (obs, reward, done, False, info) if GYMNASIUM else (obs, reward, done, info)


def reset(env):
    """observation of a new episode"""
    result = env.reset()
    return result[0] if GYMNASIUM else result


def step(env, action):
    """(obs, reward, done, info) of one step"""
    result = env.step(action)
    if GYMNASIUM:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info
    return result


def load_dqn(path, env=None):
    """
    DQN.load on the CPU. The models were saved with stable-baselines3 1.5, whose zip files hold the gym 0.21 spaces and
    the learning-rate / exploration schedules as pickled objects that stable-baselines3 2.x cannot read; with
    gymnasium they are replaced by the spaces of the environment (the network weights are loaded unchanged).
    """
    custom_objects = None
    if GYMNASIUM and env is not None:
        custom_objects = dict(observation_space=env.observation_space, action_space=env.action_space,
                              lr_schedule=lambda _: 0.0, exploration_schedule=lambda _: 0.0)
    return stable_baselines3.DQN.load(path, device="cpu", custom_objects=custom_objects)
