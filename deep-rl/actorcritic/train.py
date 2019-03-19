from collections import namedtuple

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
import gym


Transition = namedtuple('transition', ['r', 's', 'a', 'done'])


class Agent:
    def __init__(self, observation_shape, action_shape, discount_factor):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.discount_factor = discount_factor

        action_dim = np.product(np.array(self.action_shape))

        # TF Model

        # Parameters
        self.backbone = layers.Dense(4)
        self.state_value_head = layers.Dense(1)
        self.action_head = layers.Dense(action_dim)

        # Forward pass
        def critic_forward(input):
            observation_flat = layers.Flatten()(input)
            backbone_out = self.backbone(observation_flat)
            state_value_out = self.state_value_head(backbone_out)
            return state_value_out

        def actor_forward(input):
            observation_flat = layers.Flatten()(input)
            backbone_out = self.backbone(observation_flat)
            action_logit = self.action_head(backbone_out)
            action_logit = layers.Reshape(self.action_shape)(action_logit)
            action_pd = tfp.distributions.Categorical(logits=action_logit)
            return action_pd

        self.observation_input = layers.Input(shape=self.observation_shape)
        self.state_value_out = critic_forward(self.observation_input)
        self.action_pd = actor_forward(self.observation_input)

        # Critic loss
        self.reward_input = layers.Input(shape=[1])
        self.next_observation_input = layers.Input(self.observation_shape)
        self.next_state_value_out = tf.stop_gradient(
            critic_forward(self.next_observation_input))
        target = self.reward_input + self.discount_factor * self.next_state_value_out
        critic_loss = tf.losses.mean_squared_error(target, self.state_value_out)

        # Actor loss
        self.action_input = layers.Input(self.action_shape)
        action_log_prob = tf.reduce_sum(
            self.action_pd.log_prob(self.action_input))
        actor_loss = - (target - tf.stop_gradient(self.state_value_out)) \
                     * action_log_prob

        # Loss
        self.loss = critic_loss + actor_loss

        # Optimize operation
        self.optimize_op = tf.train.AdamOptimizer().minimize(self.loss)

    def action(self, observation):
        observation = np.expand_dims(np.asarray(observation), axis=0)
        return np.squeeze(
            self.action_pd.sample().eval(
                feed_dict={self.observation_input: observation}))

    def explore(self, env):
        trajectory = []

        obs = env.reset()
        done = False
        reward = 0

        while True:
            action = self.action(obs)
            trajectory.append(Transition(r=reward, s=obs, a=action, done=done))
            if done:
                break
            obs, reward, done, _ = env.step(action)

        return trajectory

    def learn(self, trajectory):
        rewards = np.vstack([t.r for t in trajectory[1:]])
        observations = np.vstack([t.s for t in trajectory])
        next_observations = observations[1:]
        observations = observations[:-1]
        actions = np.vstack([t.a for t in trajectory[:-1]])
        dones = np.vstack([t.done for t in trajectory[1:]])

        tf_dataset = tf.data.Dataset.from_tensor_slices({
            'observation': observations,
            'action': actions,
            'reward': rewards,
            'next_observation': next_observations,
            'done': dones
        })

        

        steps1 = pd.DataFrame(trajectory[:-1])[['s', 'a']]
        steps2 = pd.DataFrame(trajectory[1:])[['r', 's', 'done']] \
                   .rename(columns={'s': 's1'})
        steps = pd.concat([steps1, steps2], axis=1, sort=False)
        steps = steps.sample(frac=1.).reset_index(drop=True)

        batch_size = 8

        for batch_i in range(steps.shape[0] // batch_size):
            batch_start = batch_i * batch_size
            batch = steps.iloc[batch_start:batch_start + batch_size]
            feed_dict = {
                self.observation_input: batch[['s']].values,
                self.next_observation_input: batch[['s1']].values,
                self.reward_input: batch[['r']].values,
                self.action_input: batch[['a']].values
            }
            loss, _ = tf.get_default_session().run(
                [self.loss, self.optimize_op],
                feed_dict=feed_dict)
            print(f'batch #: {batch_i:4}, loss: {loss:}')


def main():
    with tf.Session() as sess:
        env = gym.make('CartPole-v0')
        agent = Agent(env.observation_space.shape, env.action_space.shape, 0.99)
        sess.run(tf.global_variables_initializer())

        for episode_i in range(10):
            trajectory = agent.explore(env)
            reward_sum = np.sum([t.r for t in trajectory])
            print(f'episode #: {episode_i:2}, return: {reward_sum:4}')
            agent.learn(trajectory)


if __name__ == '__main__':
    main()
