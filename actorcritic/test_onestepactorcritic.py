import gym
from . import onestepactorcritic as m

def test_denorm_action():
    env = gym.make('Pendulum-v0')
    agent = m.Agent(env.observation_space, env.action_space, 0, 0, 0)
    print(env.action_space.low, env.action_space.high)
    assert agent.denorm_action(1) == 2.
    assert agent.denorm_action(0) == 0.
    assert agent.denorm_action(-1) == -2.
