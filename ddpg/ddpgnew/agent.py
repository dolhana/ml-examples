# pylint: disable=E1129

import numpy as np
import tensorflow as tf
import gym
from .replaybuffer import ReplayBuffer
from .util import BoxSpaceNormalizer


class Agent():
    def __init__(self, env, batch_size=64):
        self.env = env
        self._batch_size = batch_size
        self.done = True
        self._replaybuffer = ReplayBuffer(10**6)
        self._build_networks()
        self.action_normer = BoxSpaceNormalizer(self.env.action_space.low, self.env.action_space.high)

        # Initialize all variables
        tf.get_default_session().run(tf.global_variables_initializer())

        # TODO Initialize target networks

    def _build_networks(self):
        self.state_0_ph = tf.placeholder(
            tf.float32, shape=[None, self.env.observation_space.shape[0]], name='state_0')
        self.mu = self._build_actor(self.state_0_ph, 'actor_online')
        self.theta_mu = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_online')

        self.state_1_ph = tf.placeholder(
            tf.float32, shape=[None, self.env.observation_space.shape[0]], name='state_1')
        self.mu_target = self._build_actor(self.state_1_ph, 'actor_target')
        self.theta_mu_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_target')

    def _build_actor(self, states_ph, scope):
        with tf.variable_scope.variable_scope(scope):
            l1 = tf.layers.dense(states_ph, 32,
                                 activation=tf.nn.relu)
            mu = tf.layers.dense(l1, self.env.action_space.shape[0],
                                 activation=tf.nn.tanh, name='mu')
        return mu


    def restart_episode(self, training=False):
        self.training = training
        self._last_observation = self.env.reset()
        self.done = False

    def step(self):
        if self.done:
            self._last_observation = self.env.reset()
        action = self.decide_action(self._last_observation)
        print(self._last_observation, action)
        observation, reward, done, _info = self.env.step(action)
        self._record_step(self._last_observation, action, reward, observation)
        self._last_observation = observation

        if self.training:
            self._train()

        return done

    def decide_action(self, observation):
        normed_action = self.mu.eval(feed_dict={self.state_ph: [observation]})[0]
        return self.action_normer.denorm(normed_action)

    def _record_step(self, observation_0, action, reward, observation_1):
        self._replaybuffer.add((observation_0, [action], [reward], observation_1))

    def _train(self):
        step_sample = self._replaybuffer.sample(self._batch_size)
        _observation_0_batch = [s[0] for s in step_sample]
        _observation_1_batch = [s[3] for s in step_sample]
        _action_batch = [s[1] for s in step_sample]
        _reward_batch = [s[2] for s in step_sample]


def main():
    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    with tf_graph.as_default(), tf_sess.as_default():
        env = gym.make('Pendulum-v0')
        agent = Agent(env)
        agent.restart_episode()

        done = False
        while not done:
            env.render()
            done = agent.step()
