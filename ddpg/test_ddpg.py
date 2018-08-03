# pylint: disable=missing-docstring,not-context-manager

from datetime import datetime
import tensorflow as tf
import gym
from ddpg.ddpg import Agent


def run(env_id, episodes, steps_per_episode, render=False):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        agent = Agent(gym.make(env_id))

    with tf_graph.as_default(), tf.Session(graph=tf_graph).as_default():
        datetime_str = datetime.now().strftime('%m%d%H%M%S')
        logdir = '/tmp/logdir/{}/{}'.format(env_id, datetime_str)
        with tf.summary.FileWriter(logdir) as summary_writer:
            _ = agent.learn(episodes, steps_per_episode,
                            summary_writer=summary_writer, render=render)

def test_pendulum():
    run('Pendulum-v0', 1, 10**9)

def test_mountaincarcontinuous():
    run('MountainCarContinuous-v0', 10**3, 10**9)

def test_tfgraph():
    run('Pendulum-v0', 0, 0)
