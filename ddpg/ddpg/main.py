import tensorflow as tf
import gym
from .ddpg import Agent


tf_graph = tf.Graph()
with tf_graph.as_default():
    agent = Agent(gym.make('MountainCarContinuous-v0'))
    
with tf_graph.as_default(), tf.Session(graph=tf_graph).as_default():
    with tf.summary.FileWriter('logdir') as summary_writer:
        avg_returns = agent.learn(1000, 1000, summary_writer=summary_writer, render=True)
