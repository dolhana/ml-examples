# pylint: unused-import
"""Demonstrates Proximal Policy Optimization algorithm

arXiv:1707.06347 [cs.LG]
https://arxiv.org/abs/1707.06347
"""
import argparse
import asyncio

import tensorflow as tf
import gym


class PPOModel():

    def __init__(self,
                 env: gym.Env,
                 ppo_epsilon=0.2,
                 actor_lr=1e-4,
                 critic_lr=1e-4,
                 activation=tf.nn.tanh,
                 tf_graph=None):
        if tf_graph is None:
            tf_graph = tf.get_default_graph()

        self.env = env
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape

        self.state_ph = tf.placeholder(
            tf.float32, shape=[None, *self.state_shape], name='state')

        with tf.variable_scope('actor'):
            l1 = tf.layers.dense(self.state_ph, 64, activation=activation)
            l2 = tf.layers.dense(l1, 64, activation=activation)
            action_mean = tf.layers.dense(l2, self.action_shape[0],
                                          activation=None, name='action_mean')
            action_logstd = tf.get_variable('action_logstd',
                                            shape=self.action_shape[0], dtype=tf.float32,
                                            initializer=tf.initializers.zeros())
            action_std = tf.exp(action_logstd, name='action_stddev')
            self.action_pd = tf.distributions.Normal(
                action_mean, action_std, name='action_pd')
            self.action_out = self.action_pd.sample()

        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.state_ph, 64, activation=activation)
            l2 = tf.layers.dense(l1, 64, activation=activation)
            self.value_out = tf.dense(l2, 1, activation=None)

        with tf.variable_scope('train'):
            self.advantage_ph = tf.placeholder(tf.float32, name='advantage')
            self.action_ph = tf.placeholder(
                tf.float32, shape=[None, self.action_shape[0]], name='action')
            action_prob = self.action_pd.prob(self.action_ph)
            action_prob_old = tf.stop_gradient(action_prob,
                                               name='action_prob_old')
            r = action_prob / action_prob_old
            actor_obj = tf.reduce_mean(tf.minimum(
                r * self.advantage_ph,
                tf.clip_by_value(r, 1 - ppo_epsilon, 1 + ppo_epsilon) * self.advantage_ph))
            critic_obj = tf.reduce_mean(self.advantage_ph * self.value_out)

            self.actor_optimizer = \
                tf.train.AdamOptimizer(learning_rate=actor_lr) \
                        .minimize(-actor_obj)
            self.critic_optimizer = \
                tf.train.AdamOptimizer(learning_rate=critic_lr) \
                        .minimize(-critic_obj)

    def choose_action(self, state, tf_session=None):
        if tf_session is None:
            tf_session = tf.get_default_session()
        assert tf_session is not None

        action = tf_session.run(self.action_out, feed_dict={self.state_ph: [state]})[0]
        return action

    def train_minibatch(self, minibatch, tf_session=None):
        """Trains with a minibatch

        Args:
        	minibatch: [(state, action, advantage)]
        """
        if tf_session is None:
            tf_session = tf.get_default_session()
        states, actions, advantages = zip(*minibatch) if minibatch else [[], [], []]

        tf_session.run(
            [self.critic_optimizer, self.actor_optimizer],
            feed_dict={
                self.state_ph: states,
                self.action_ph: actions,
                self.advantage_ph: advantages
            })


def build_argparser():
    argparser = argparse.ArgumentParser(description='Demonstrates Proximal Policy Optimization')
    argparser.add_argument(
        '--env-id', help='OpenAI gym environment ID; defaults to "Pendulum-v0"',
        dest='env_id', default='Pendulum-v0')
    argparser.add_argument(
        '--num-agents', help='The number of agents that explore the environment simulateneously',
        dest='n_agents', default=1)
    argparser.add_argument(
        '--horizon', help='The maximum number of steps that each agent can take in each iteration',
        dest='horizon', default=200)
    argparser.add_argument(
        '--num-iterations', help='The number of iterations',
        dest='n_iterations', default=1)
    return argparser


def make_policy(env_id):
    env = gym.make(env_id)
    return lambda _state: env.action_space.sample()

def make_agent(env_id, policy):
    class Agent:
        def __init__(self, env_id, policy):
            self.env = gym.make(env_id)
            self.policy = policy

        def __call__(self, horizon):
            return self.explore(horizon)

        def explore(self, horizon):
            trajectory = []
            done = False
            for step_i in range(horizon):

def main():
    """Demonstrates Proximal Policy Optimization"""
    argparser = build_argparser()
    args = argparser.parse_args()

    loop = asyncio.get_event_loop()

    # PPO algorithm, actor-critic style

    # Make the model
    policy = make_policy(args.env_id)

    agents = [make_agent(args.env_id, policy) for _ in range(args.n_agents)]

    for i_iter in range(args.n_iterations):
        trajectories_future = asyncio.gather()
        loop.run_until_complete(trajectories_future)

if __name__ == '__main__':
    main()
