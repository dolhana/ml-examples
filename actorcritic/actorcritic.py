# pylint: disable=E1129

import tempfile
import os
import tensorflow as tf
import gym


class Agent():
    def __init__(self, env,
                 gamma, lambda_,
                 critic_lr, actor_lr):
        self.env = env
        self.gamma = gamma
        self.lambda_ = lambda_
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        self.state_ph = tf.placeholder(
            tf.float32, shape=[None] + list(self.env.observation_space.shape), name='state')
        self.actor = ActorModel(self.env.observation_space.shape, self.env.action_space.shape)
        self.critic = CriticModel(self.env.observation_space.shape)

        with tf.variable_scope('loss'):
            self.action_ph = tf.placeholder(tf.float32, shape=[None, self.env.action_space.shape[0]], name='action')
            self.advantage_ph = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')
            self.critic_loss = - tf.reduce_mean(self.critic.value_out * self.advantage_ph)
            self.actor_loss = - tf.reduce_mean(self.actor.pd.log_prob(self.action_ph) * self.advantage_ph)
            self.loss_summary = tf.summary.merge([
                tf.summary.scalar('critic_loss', self.critic_loss),
                tf.summary.scalar('actor_loss', self.actor_loss)])

        with tf.variable_scope('optimizer'):
            self.critic_optimizer = \
                tf.train.AdamOptimizer(learning_rate=critic_lr) \
                .minimize(self.critic_loss)
            self.actor_optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.actor_lr) \
                .minimize(self.actor_loss)
    
    def train(self, n_iterations, t_max, n_epochs, batch_size, summary_writer):
        obs0 = self.env.reset()
        done = False

        for _iter_i in range(n_iterations):
            advs = []
            states = []
            actions = []

            for t in range(t_max):
                action = self.actor.actions([obs0])[0]
                obs1, reward, done, _info = self.env.step(action)
                self.env.render()

                states.append(obs0)
                actions.append(action)
                td_residual = reward + self.gamma * self.critic.values([obs1])[0] - self.critic.values([obs0])[0]
                advs.append(td_residual)
                for i in range(-2, -2 - t, -1):
                    td_residual *= self.gamma * self.lambda_
                    advs[i] += td_residual
                
                if done:
                    obs0 = self.env.reset()
                    done = False
                    break

                obs0 = obs1

            feed_dict = {
                self.critic.state_ph: states,
                self.actor.state_ph: states,
                self.advantage_ph: advs,
                self.action_ph: actions
            }

            if not summary_writer is None:
                loss_summary = tf.get_default_session().run(
                    self.loss_summary, feed_dict=feed_dict
                )
                summary_writer.add_summary(loss_summary, global_step=_iter_i)

            tf.get_default_session().run(
                [self.critic_optimizer, self.actor_optimizer], feed_dict=feed_dict
            )
            


class ActorModel():
    def __init__(self, state_shape, action_shape):
        with tf.variable_scope('actor'):
            self.state_ph = tf.placeholder(
                tf.float32, shape=[None] + list(state_shape), name='state')
            l1 = tf.layers.dense(self.state_ph, 64, activation=tf.nn.tanh)
            l2 = tf.layers.dense(l1, 64, activation=tf.nn.tanh)
            pd_mean = tf.layers.dense(l2, action_shape[0], activation=None, name='pd_mean')
            pd_logstd = tf.get_variable('pd_logstd',
                shape=[action_shape[0]], dtype=tf.float32,
                initializer=tf.initializers.zeros())
            self.pd = tf.distributions.Normal(pd_mean, tf.exp(pd_logstd))
            self.out = self.pd.sample()

    def actions(self, states):
        return self.out.eval(feed_dict={self.state_ph: states})


class CriticModel():
    def __init__(self, state_shape, reuse_variables=False):
        with tf.variable_scope('critic', reuse=reuse_variables):
            self.state_ph = tf.placeholder(
                dtype=tf.float32, shape=[None] + list(state_shape), name='state')
            l1 = tf.layers.dense(self.state_ph, 64, activation=tf.nn.tanh)
            l2 = tf.layers.dense(l1, 64, activation=tf.nn.tanh)
            self.value_out = tf.layers.dense(l2, 1, activation=None)

    def values(self, states):
        return self.value_out.eval(feed_dict={self.state_ph: states})


def main():
    logdir = 'logdir'
    os.makedirs(logdir, exist_ok=True)
    run_logdir = tempfile.mkdtemp(dir=logdir)

    env = gym.make('Pendulum-v0')
    monitored_env = gym.wrappers.Monitor(env, os.path.join(run_logdir, 'gym'), force=True)

    tf_graph = tf.Graph()
    with tf_graph.as_default():
        agent = Agent(env, gamma=0.99, lambda_=0.95, critic_lr=1e-4, actor_lr=1e-4)

    # TB: Setup a summary FileWriter
    summary_writer = tf.summary.FileWriter(os.path.join(run_logdir, 'tb'), tf_graph)
    print('tensorboard logdir:', logdir)

    tf.InteractiveSession(graph=tf_graph)
    tf.global_variables_initializer().run()

    agent.train(10000, 10, 1, 1, summary_writer)

    summary_writer.close()

if __name__ == '__main__':
    main()
