import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp


class Agent:
    def __init__(self, state_shape, action_shape, discount_rate):
        self.critic = Critic(state_shape, discount_rate)
        self.actor = Actor(state_shape, action_shape)

    def learn(self, state, action, reward, next_state, done):
        critic_loss = self.critic.fit(state, reward, next_state, done)
        actor_loss = self.actor.fit(state, reward, next_state, done)
        return {'critic_loss': critic_loss, 'actor_loss': actor_loss}


class Critic:
    def __init__(self, state_shape, discount_rate):
        self.discount_rate = discount_rate
        self.input = layers.Input(shape=state_shape)
        x = layers.Flatten()(self.input)
        self.output = layers.Dense(1)(x)

        self.target = layers.Input(shape=[1])
        self.loss = tf.losses.mean_squared_error(
            labels=self.target,
            predictions=self.output
        )
        self.optimize = tf.train.AdamOptimizer().minimize(self.loss)

    def __call__(self, state):
        return self.output.eval(feed_dict={self.input: state})

    def learn(self, state, reward, next_state, done):
        target = reward + self.discount_rate * self(state)
        


class Actor:
    def __init__(self, state_shape, action_shape):
        output_dim = np.product(action_shape)
        self.input = layers.Input(shape=state_shape)
        x = layers.Flatten()(self.input)
        logits = layers.Dense(output_dim)(x)
        logits = layers.Reshape(action_shape)(logits)
        self.pd = tfp.distributions.Categorical(logits=logits)

    def __call__(self, state):
        return self.pd.eval(feed_dict={self.input: state})
