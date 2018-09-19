import gym
import tensorflow as tf


class Agent():
    def __init__(self, observation_space, action_space, actor_lr, critic_lr, discount_rate):
        self.observation_space = observation_space
        self.action_space = action_space
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_rate = discount_rate

        def build_critic(state, reuse=False):
            with tf.variable_scope('critic', reuse=reuse):
                l = state
                l = tf.layers.dense(l, 64, activation=tf.nn.tanh)
                l = tf.layers.dense(l, 64, activation=tf.nn.tanh)
                return tf.layers.dense(l, 1, activation=None)

        self.state_ph = tf.placeholder(
            tf.float32, shape=[None] + list(self.observation_space.shape), name='s')

        # actor network
        with tf.variable_scope('actor'):
            l = self.state_ph
            l = tf.layers.dense(l, 64, activation=tf.nn.tanh)
            l = tf.layers.dense(l, 64, activation=tf.nn.tanh)
            action_mean = tf.layers.dense(l, action_space.shape[0], activation=None, name='mean')
            action_logstd = tf.get_variable('logstd', shape=[action_space.shape[0]], dtype=tf.float32, initializer=tf.initializers.zeros())
            action_std = tf.exp(action_logstd)
            self.action_pd = tf.distributions.Normal(action_mean, action_std)
            self.action_out = self.action_pd.sample(name='action')

        self.value_out = build_critic(self.state_ph)
        
        # losses
        self.next_state_ph = tf.placeholder(
            tf.float32, shape=[None] + list(self.observation_space.shape), name='s1')
        self.reward_ph = tf.placeholder(
            tf.float32, shape=[None, 1], name='r')
        self.next_value_out = build_critic(self.next_state_ph, reuse=True)
        self.advantage = self.reward_ph + self.discount_rate * self.next_value_out - self.value_out

        self.action_ph = tf.placeholder(tf.float32, shape=[None, action_space.shape[0]], name='action')
        self.advantage_ph = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')
        self.critic_loss = - tf.reduce_mean(self.value_out * self.advantage_ph)
        self.actor_loss = - tf.reduce_mean(self.action_pd.log_prob(self.action_ph) * self.advantage_ph)

        # optimizers
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(self.critic_loss)
        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr).minimize(self.actor_loss)

        self.reset_episode()

    def reset_episode(self):
        self.last_state = None
        self.last_action = None
        self.done = False
        self.episode_return = 0
        self.episode_steps = 0

    def decide_action(self, observation):
        if self.done:
            self.reset_episode()
        normed_action = self.action_out.eval(feed_dict={self.state_ph: [observation]})[0]
        self.last_state = observation
        self.last_action = normed_action
        return self.denorm_action(self.last_action)

    def observe(self, observation, reward, done):
        self.episode_return += reward
        self.episode_steps += 1

        # train
        _value_out_val, _next_value_out_val, advantage_val = tf.get_default_session().run(
            [self.value_out, self.next_value_out, self.advantage],
            feed_dict={
                self.state_ph: [self.last_state],
                self.reward_ph: [[reward]],
                self.next_state_ph: [observation]
            })
        #print([self.last_state, [reward], observation, value_out_val, next_value_out_val, advantage_val])
        critic_loss_val, actor_loss_val = tf.get_default_session().run(
            [self.critic_loss, self.actor_loss],
            feed_dict={
                self.state_ph: [self.last_state],
                self.action_ph: [self.last_action],
                self.advantage_ph: advantage_val
            })
        print([critic_loss_val, actor_loss_val, self.episode_return / self.episode_steps, self.last_action])
        tf.get_default_session().run(
            [self.critic_optimizer, self.actor_optimizer],
            feed_dict={
                self.state_ph: [self.last_state],
                #self.reward_ph: [[reward]],
                self.action_ph: [self.last_action],
                self.advantage_ph: advantage_val
                #self.next_state_ph: [observation]
            })

        self.last_state = observation
        self.done = done

    def denorm_action(self, normed_action):
        return (normed_action + 1) * (self.action_space.high - self.action_space.low) / 2 + self.action_space.low


def main():
    tf.InteractiveSession()
    env = gym.make('Pendulum-v0')
    agent = Agent(env.observation_space, env.action_space, 1e-4, 1e-4, 0.99)
    tf.global_variables_initializer().run()

    for episode in range(10000):
        observation = env.reset()
        agent.reset_episode()
        for _step in range(1000):
            env.render()
            action = agent.decide_action(observation)
            observation_next, reward, done, _info = env.step(action)
            print(observation, action, observation_next, reward, done)
            agent.observe(observation_next, reward, done)
            observation = observation_next
            if done:
                print('episode {} ended.'.format(episode))
                break


if __name__ == '__main__':
    main()
