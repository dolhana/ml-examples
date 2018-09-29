from typing import Tuple, Callable
import tensorflow as tf
import gym


class Agent():
    """OpenAI gym agent"""

    def __init__(self, env, model_builder):
        """Creates an agent that interacts with the given env"""
        self.env = env
        self.last_state = None
        self.done = True
        self.model = model_builder(env)

    def step(self, auto_restart_episode=True):
        """agent.step() runs one step in the environment

        Args:
	        auto_restart_episode (bool):
        				If True, resets the env and runs a step if self.done == True.
        				If False, tries to run a step even if self.done == True.

        Returns:
        	(prev_state, action, reward, cur_state, done, info)
        """
        if self.done and auto_restart_episode:
            self.restart_episode()

        action = self.choose_action(self.last_state)
        state, reward, done, _info = self.env.step(action)

        return self.last_state, action, reward, state, done, _info

    def restart_episode(self):
        """Resets the environment to start a new episode

        Returns:
        	state:	a new state
        """
        self.last_state = self.env.reset()
        self.done = False

    def choose_action(self, state):
        """Choose an action based on the state

        Args:
        	state:	agent chooses an action based on this state

        Returns:
        	action
        """
        action = self.model.choose_action(self.last_state)
        return action


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
