import tqdm
import tensorflow as tf
import tensorflow.keras.layers as layers
import gym
from util import *

class DiscretePolicy:
    def __init__(self, n_action_classes, state_shape, alpha=1e-3, tf_sess=None):
        self.n_action_classes = n_action_classes
        self.state_shape = state_shape
        self.alpha = alpha
        self.tf_sess = tf_sess if tf_sess is not None else tf.get_default_session()

        # Define layers
        self.state = layers.Input(shape=state_shape)
        self.hidden_layer = layers.Dense(4, activation=tf.tanh)
        self.logits_layer = layers.Dense(self.n_action_classes)
        self.layers = [self.hidden_layer, self.logits_layer]
        
        # Fowrad propagation
        x = self.hidden_layer(self.state)
        logits = self.logits_layer(x)
        self.pd = tf.distributions.Categorical(logits)

        # Define the model
        self.model = tf.keras.Model(inputs=self.state, outputs=logits)

        # Running `predict` initializes the variables
        self.model.predict(np.zeros(shape=(1,) + state_shape))

        # Define the objective function
        self.action = layers.Input(shape=(1,))
        self.value = layers.Input(shape=(1,))
        self.obj_fn = tf.reduce_mean(self.pd.log_prob(self.action) * self.value)

        # Create an optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        self.optimize_op = self.optimizer.minimize(- self.obj_fn)
        
    def sample(self, state):
        return self.tf_sess.run(self.pd.sample(), feed_dict={self.state: state})

    def train(self, state, action, value):
        self.tf_sess.run(self.optimize_op, feed_dict={
            self.state: state,
            self.action: action,
            self.value: value
        })


def train(epochs=1, alpha=1e-3, gamma=1., normalize_return=True, render=False):
    tf_sess = tf.Session()
    env = gym.make('CartPole-v0')
    policy = DiscretePolicy(
        env.action_space.n,
        env.observation_space.shape,
        alpha=alpha, tf_sess=tf_sess)
    print(policy.model.summary())

    tf_sess.run(tf.global_variables_initializer())

    def policy_action(obs):
        return policy.sample(obs[None, :]).squeeze()

    returns = []
    emav = ExponentialMovingAverageVariance(shape=(1,), alpha=0.01)

    pbar = tqdm.trange(epochs)
    for _ in pbar:
        trajectory = collect_episode(env, policy_action, render=render)
        states = np.vstack([e.state for e in trajectory])
        actions = np.vstack([e.action for e in trajectory])
        rewards = np.vstack([e.reward for e in trajectory])

        returns.append(np.sum(rewards))
        pbar.set_postfix({'return': returns[-1]})

        discounted_returns = calc_discounted_returns(rewards, gamma)

        if normalize_return:
            for x in discounted_returns.squeeze():
                emav.update(x)
            discounted_returns = (discounted_returns - emav.mean) / emav.stddev()

        policy.train(states, actions, discounted_returns)

    return returns
