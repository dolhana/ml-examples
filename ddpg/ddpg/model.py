import numpy as np
import tensorflow as tf
from gym import spaces


class Model():

    def __init__(self, observation_dim: int, action_dim: int,
                 tau=0.001, gamma=0.99, critic_l2_rate=0.01,
                 critic_learning_rate=0.0001, actor_learning_rate=0.00001):

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.tau = tau
        self.gamma = gamma
        self.critic_l2_rate = critic_l2_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate

        self.observations = tf.placeholder(
            tf.float32, shape=[None, self.observation_dim], name='observations')
        self.observations_next = tf.placeholder(
            tf.float32, shape=[None, self.observation_dim], name='observations_next')
        self.actions = tf.placeholder(
            tf.float32, shape=[None, self.action_dim], name='actions')
        self.rewards = tf.placeholder(
            tf.float32, shape=[None, 1], name='rewards')

        self.critic_training = tf.placeholder(tf.bool, name='critic_training')
        self.actor_training = tf.placeholder(tf.bool, name='actor_training')

        with tf.variable_scope('normalized_observation'):
            self.normalized_observation = tf.layers.batch_normalization(
                inputs=tf.identity(self.observations),
                training=self.critic_training,
                name='normalize_observation'
            )
            self.normalized_observation_next = tf.layers.batch_normalization(
                inputs=tf.identity(self.observations_next),
                training=self.critic_training, reuse=True,
                name='normalize_observation'
            )

        self.critic = Critic(
            self.normalized_observation, self.actions, training=self.critic_training)
        self.actor = Actor(
            self.action_dim, self.normalized_observation, training=self.actor_training)

        self.actor_target = Actor(
            self.action_dim, self.normalized_observation_next,
            training=False, name='actor_target')
        self.critic_target = Critic(
            self.normalized_observation_next, self.actor_target.mu,
            training=False, name='critic_target')

        # Set y_i = r_i + gamma * target_q(s_{i+1}, target_mu(s_{i+1}))
        y = self.rewards + self.gamma * self.critic_target.q

        # Update critic by minimizing the loss L = mse( y_i - q(s_i, a_i) )
        self.critic_loss = tf.add(
            tf.losses.mean_squared_error(y, self.critic.q),
            tf.multiply(
                self.critic_l2_rate,
                tf.losses.get_regularization_loss(
                    scope=self.critic.name,
                    name='critic_regularization_loss')))
        tf.summary.scalar('critic_loss', self.critic_loss)

        batchnorm_update_ops = (
            tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.critic.name)
            + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='normalized_observation')
        )
        with tf.control_dependencies(batchnorm_update_ops):
            var_list = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.critic.name)
            self.critic_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.critic_learning_rate).minimize(
                    self.critic_loss, var_list=var_list)

        # Update the actor policy using the sampled policy gradient
        #   J = mean( q(s_i, mu(s_i)) )
        # d_J = mean( d_q(s_i, mu(s_i)) * d_mu(s_i) )
        self.critic_with_actor = Critic(
            self.normalized_observation, self.actor.mu, training=False, reuse=True)
        self.actor_loss = - tf.reduce_mean(self.critic_with_actor.q)
        tf.summary.scalar('actor_loss', self.actor_loss)

        batchnorm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.actor.name)
        with tf.control_dependencies(batchnorm_update_ops):
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.actor.name)
            self.actor_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.actor_learning_rate).minimize(
                    self.actor_loss, var_list=var_list)

        # Update the target networks
        # target_theta_q  = tau * theta_q  + (1 - tau) * target_theta_q
        # target_theta_mu = tau * theta_mu + (1 - tau) * target_theta_mu
        def target_update_ops(target, origin, tau):
            target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope=target.name)
            init_ops = []
            update_ops = []

            for target_var in target_vars:
                origin_var_name = target_var.name.replace(target.name + '/', origin.name + '/', 1)
                origin_var = tf.get_default_graph().get_tensor_by_name(origin_var_name)
                init_ops.append(
                    tf.assign(target_var, origin_var))
                update_ops.append(
                    tf.assign(target_var, (1 - tau) * target_var + tau * origin_var))
            return tf.group(*init_ops), tf.group(*update_ops)

        with tf.variable_scope('critic_target'):
            self.critic_target_init, self.critic_target_update = \
                target_update_ops(self.critic_target, self.critic, self.tau)
        with tf.variable_scope('actor_target'):
            self.actor_target_init, self.actor_target_update = \
                target_update_ops(self.actor_target, self.actor, self.tau)

        self.summary_merged = tf.summary.merge_all()

    def reset(self):
        tf.global_variables_initializer().run(session=tf.get_default_session())

        # Initialize target network target_q and target_mu with weights target_theta_q and target_mu_q
        # and copy (theta_q, theta_mu) to (target_theta_q, target_mu_q)
        tf.get_default_session().run([
            self.critic_target_init, self.actor_target_init
        ])

    def mu(self, observations: np.ndarray) -> np.ndarray:
        return self.actor.mu.eval(feed_dict={
            self.observations: observations,
            self.actor_training: False,
            self.critic_training: False})

    def q(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.critic.q.eval(feed_dict={
            self.observations: observations, self.actions: actions,
            self.critic_training: False})

    def learn(self, observations, actions, rewards, observations_next, dones):
        self.critic_optimizer.run(feed_dict={
            self.observations: observations,
            self.actions: actions,
            self.rewards: rewards,
            self.observations_next: observations_next,
            self.critic_training: True
        })
        self.actor_optimizer.run(feed_dict={
            self.observations: observations,
            self.actor_training: True,
            self.critic_training: False
        })
        tf.get_default_session().run([
            self.critic_target_update,
            self.actor_target_update
        ])

    def eval_summary_merged(self, observations, actions, rewards, observations_next, dones):
        return self.summary_merged.eval(feed_dict={
            self.observations: observations,
            self.actions: actions,
            self.rewards: rewards,
            self.observations_next: observations_next,
            self.critic_training: False,
            self.actor_training: False
        })

    def target_softupdate(self, tau):
        """Update the target networks

        target_theta_q  = tau * theta_q  + (1 - tau) * target_theta_q
        target_theta_mu = tau * theta_mu + (1 - tau) * target_theta_mu
        """
        tf.get_default_session().run(
            [self.critic_target_update, self.actor_target_update],
            feed_dict={ self.tau: tau }
        )


class Critic():
    def __init__(self, observations: tf.Tensor, actions: tf.Tensor, training, reuse=False, name: str = 'critic'):
        self.name = name
        self.observations = observations
        self.actions = actions

        tf_regularizer = tf.contrib.layers.l2_regularizer(scale=1.)

        def states_hidden_layer(inputs: tf.Tensor, units: int):
            net = tf.layers.dense(
                inputs, units=units, use_bias=False,
                kernel_initializer=hidden_layer_initializer(inputs),
                kernel_regularizer=tf_regularizer)
            net = tf.layers.batch_normalization(net, training=training)
            net = tf.nn.relu(net)
            return net

        def concat_hidden_layer(inputs: tf.Tensor, units: int):
            net = tf.layers.dense(
                inputs, units=units, use_bias=False,
                kernel_initializer=hidden_layer_initializer(inputs),
                kernel_regularizer=tf_regularizer)
            net = tf.layers.batch_normalization(net, training=training)
            net = tf.nn.relu(net)
            return net

        def final_layer(inputs: tf.Tensor):
            net = tf.layers.dense(
                inputs, units=1, use_bias=True,
                kernel_initializer=final_layer_initializer(),
                bias_initializer=final_layer_initializer(),
                kernel_regularizer=tf_regularizer)
            return net

        scope_reuse = True if reuse else tf.AUTO_REUSE
        with tf.variable_scope(self.name, reuse=scope_reuse):
            states_net = self.observations
            states_net = states_hidden_layer(states_net, 256)
            states_net = states_hidden_layer(states_net, 256)

            actions_net = self.actions

            net = tf.concat(values=[states_net, actions_net], axis=-1)
            net = concat_hidden_layer(net, 256)

            net = final_layer(net)
            self.q = tf.identity(net, name='q')


class Actor():
    def __init__(self, action_dim: int, observations: tf.Tensor, training, name='actor'):
        self.name = name
        self.action_dim = action_dim
        self.observations = observations

        def hidden_layer(inputs: tf.Tensor, units: int):
            net = tf.layers.dense(inputs, units=units, use_bias=False,
                                  kernel_initializer=hidden_layer_initializer(inputs))
            net = tf.layers.batch_normalization(net, training=training)
            return tf.nn.relu(net)

        def final_layer(inputs: tf.Tensor):
            return tf.layers.dense(inputs, units=self.action_dim,
                                   kernel_initializer=final_layer_initializer())

        with tf.variable_scope(self.name):
            net = self.observations
            net = hidden_layer(net, units=256)
            net = final_layer(net)
            net = tf.nn.tanh(net)
            self.mu = tf.identity(net, name='mu')


def hidden_layer_initializer(inputs: tf.Tensor):
    stddev = .5 * inputs.shape[1].value ** -.5
    return tf.initializers.truncated_normal(stddev=stddev)

def final_layer_initializer():
    stddev = .5 * 3.e-3
    return tf.initializers.truncated_normal(stddev=stddev)
