# pylint: disable=missing-docstring
import os

import numpy as np
import tensorflow as tf
import gym

from train import * # pylint: disable=unused-wildcard-import,wildcard-import


os.environ['CUDA_VISIBLE_DEVCIES'] = ''

def test_train():
    with tf.Session(config=tf.ConfigProto(
            device_count={
                'GPU': 0
            }
    )) as sess:
        env = gym.make('CartPole-v0')
        agent = Agent(env.observation_space.shape, env.action_space.shape, 0.99)
        sess.run(tf.global_variables_initializer())

        for episode_i in range(10):
            trajectory = agent.explore(env)
            reward_sum = np.sum([t.r for t in trajectory])
            print(f'episode #: {episode_i:2}, return: {reward_sum:4}')
            agent.learn(trajectory)


def test_env():
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    print(f'environment name: {env_name}')
    print(f'observation space: {env.observation_space}')
    print(f'action space: {env.action_space}')

    # one episode exploration
    done = False
    reward_sum = 0

    env.reset()
    while not done:
        action = env.action_space.sample()
        _obs, reward, done, _ = env.step(action)
        reward_sum += reward

    print(f'one episode random exploration return: {reward_sum}')
