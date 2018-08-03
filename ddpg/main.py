from datetime import datetime
import tensorflow as tf
import gym
from ddpg.ddpg import Agent


def run(env_id):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        agent = Agent(gym.make(env_id))

    with tf_graph.as_default(), tf.Session(graph=tf_graph).as_default():
        datetime_str = datetime.now().strftime('%m%d%H%M%S')
        with tf.summary.FileWriter('/tmp/logdir/{}/{}'.format(env_id, datetime_str)) as summary_writer:
            avg_returns = agent.learn(10**3, 10**8, summary_writer=summary_writer, render=True)

if __name__ == '__main__':
    run('Pendulum-v0')
    # run('MountainCarContinuous-v0')
