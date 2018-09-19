import numpy as np
import gym
import tensorflow as tf

from .model import Model
from .replaybuffer import ReplayBuffer
from .noise import OUNoise


class Agent():
    """"""

    def __init__(self, env: gym.Env):
        self.env = env

        self.model = Model(self.env.observation_space.shape[0],
                           self.env.action_space.shape[0],
                           critic_learning_rate=0.001,
                           actor_learning_rate=0.001)

    def learn(self, num_episode, num_steps_per_episode,
              noise_theta=0.15, noise_sigma=0.2,
              replay_buffer_size=10**6, batch_size=256,
              summary_writer=None, render=False):
        """DDPG algorithm implementation"""

        # Randomly initialize critic network q(s, a|theta_q) and actor mu(s|theta_mu)
        # with weights theta_q and theta_mu
        self.model.reset()

        # Initialize the replay buffer R
        replaybuffer = ReplayBuffer(
            buffer_size=replay_buffer_size, batch_size=batch_size)

        # Track the performance
        episode_returns = []
        global_step_count = 0

        for _episode in range(1, num_episode+1):
            # Initailzie the noise process N for action exploration
            noise_process = OUNoise(
                size=self.env.action_space.shape[0],
                mu=0, theta=noise_theta, sigma=noise_sigma
            )

            # Receive initial observation state s1
            observation = self.env.reset()
            if render:
                self.env.render()

            for _t in range(1, num_steps_per_episode+1):
                global_step_count += 1

                # Select an action according to the currenty policy and exploration noise
                # a_t = mu(s_t) + N_t
                normalized_action = self.model.mu(observation.reshape(1, -1)) \
                                    + noise_process.sample()
                action = self.denorm_action(normalized_action)

                # Execute action a_t and observe reward r_t and new state s_(t+1)
                observation_next, reward, done, _info = self.env.step(action[0])

                if render:
                    self.env.render()

                # Store transition (s_t, a_t, r_t, s_(t+1)) in R
                replaybuffer.add(observation, normalized_action, reward,
                                 observation_next, done)

                # Sample a random minibatch of N transitions (s_i, a_i, r_i, s_(i+1)) from R
                batch = replaybuffer.sample()

                if len(batch) > 0:
                    # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
                    observations = np.vstack([e.observation for e in batch])
                    actions = np.vstack([e.action for e in batch]).astype(np.float32)
                    rewards = np.vstack([e.reward for e in batch]).astype(np.float32)
                    dones = np.vstack([e.done for e in batch]).astype(np.uint8)
                    observations_next = np.vstack([e.observation_next for e in batch])

                    if not summary_writer is None:
                        summary = self.model.eval_summary_merged(
                            observations, actions, rewards, observations_next, dones)
                        summary_writer.add_summary(summary, global_step_count)
                        print('summary written', global_step_count)

                    self.model.learn(observations, actions, rewards, observations_next, dones)

                if done:
                    break

            # evaluate the performance
            # episode_return = np.sum(self.evaluate(render=render))
            # print('{}\t{}'.format(_episode, episode_return))
            # episode_returns.append(episode_return)

        return episode_returns

    def evaluate(self, render=False):
        rewards = []
        done = False
        observation = self.env.reset()

        if render:
            self.env.render()

        while not done:
            action = self.model.mu(observation.reshape((1, -1)))[0]
            action = self.denorm_action(action)
            observation_next, reward, done, _info = self.env.step(action)

            if render:
                self.env.render()

            rewards.append(reward)
            observation = observation_next
        return rewards

    def denorm_action(self, norm_action):
        action_radius = (self.env.action_space.high - self.env.action_space.low) / 2.
        action_mid = self.env.action_space.low + action_radius
        return norm_action * action_radius + action_mid

    def save_model(self, summary_writer):
        summary_writer.add_graph(tf.get_default_graph())
