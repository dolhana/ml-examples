from collections import namedtuple
import functools, operator

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as layers
import gym


Transition = namedtuple('transition', ['r', 's', 'a', 'done'])

def onestep_actorcritic(agent, make_env, num_iterations, actor_step_size, critic_step_size, max_time_steps):
    env = make_env()
    for _ in range(num_iterations):
        state = env.reset()

        for _ in range(max_time_steps):
            action = agent.step(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            if done:
                break
            state = next_state

class Critic:
    def __init__(self, state_shape):
        # Parameters
        self.output_layer = layers.Dense(1)

        # Eval graph
        self.input = layers.Input(shape=state_shape)
        input_flat = layers.Flatten(self.input)
        self.output = self.output_layer(input_flat)

        # Train graph

        # Hyperparams
        self.gamma = tf.placeholder(dtype=tf.float32, shape=[1])
        self.step_size = tf.placeholder(dtype=tf.float32, shape=[1])

        # Inputs
        self.input_s0 = layers.Input(shape=state_shape)
        input_s0_flat = layers.Flatten()(self.input_s0)
        self.input_s1 = layers.Input(shape=state_shape)
        input_s1_flat = layers.Flatten()(self.input_s1)
        input_s1_flat = tf.reshape(self.input_s1, shape=[-1, input_dim])
        self.input_reward = layers.Input(shape=[1])

        # Loss
        s0_value = self.output_layer(input_s0_flat)
        s1_value = self.output_layer(input_s1_flat)
        target = tf.stop_gradient(self.input_reward + self.gamma * s1_value)
        self.loss = tf.losses.mean_squared_error(labels=target, predictions=s0_value)

        # Optimizer
        self.optimier = tf.train.AdamOptimizer(learning_rate=self.step_size).minimize(self.loss)

    def __call__(self, state):
        if not isinstance(state, tf.Tensor):
            state = tf.constant(state)
        if not state.shape.is_compatible_with(self.input.shape):
            state = tf.reshape(shape=[1, *state.shape])
        return self.output.eval(feed_dict={self.input: state})

    def fit(self, state_0, reward, state_1, discount_rate, step_size):
        feed_dict = {
            self.input_s0: state_0,
            self.input_s1: state_1,
            self.input_reward: reward,
            self.gamma: discount_rate,
            self.step_size: step_size
        }
        loss_before, _, loss_after = tf.get_default_session().run([self.loss, self.optimier, self.loss], feed_dict=feed_dict)
        return loss_before, loss_after

class Actor:
    def __init__(self, state_shape, action_shape):
        # The dimension of the output
        action_dim = np.prod(np.array(action_shape))

        # Parameters
        self.output_layer = layers.Dense(action_dim)

        # Eval graph
        self.input_state = layers.Input(shape=state_shape)
        input_state_flat = layers.Flatten()(self.input_state)
        logits = self.output_layer(input_state_flat)
        logits = layers.Reshape(action_shape)(logits)
        self.pd = tfp.Distributions.Categorical(logits=logits)

    def __call__(self, state):
        if not isinstance(state, tf.Tensor):
            state = tf.constant(state)
        is_batch = True
        if not state.shape.is_compatible_with(self.input.shape):
            is_batch = False
            state = tf.reshape(shape=[1, *state.shape])
        action = self.pd.eval(feed_dict={self.input_state: state})
        if not is_batch:
            assert action.shape[0] == 1
            action = tf.reshape(action, shape=action.shape[1:])
        return action

    def fit(self, )

class ActorCriticAgent:
    def __init__(self, observation_space, action_space):
        self.critic = Critic(observation_space)
        self.actor = Actor(observation_space, action_space)
        
    def step(self, observation):

def collect_trajectories(make_env, agent, max_time_steps):
    pass

def main():
    make_env = lambda: gym.make('CartPole-v0')
    env = make_env()
    agent = ActorCriticAgent(env.observation_space, env.state_space)

    onestep_actorcritic(agent, make_env, 10, 1e-3, 1e-3, 1000)

if __name__ == '__main__':
    main()
