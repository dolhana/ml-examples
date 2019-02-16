from collections import namedtuple
import numpy as np

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


def calc_discounted_returns(rewards: np.ndarray, discount_rate: float):
    """Calculates `gamma^t` times the discounted return `G_t` after time `t`

    The returned values are used to update policy parameter `theta`.

      d(theta) = alpha * (gamma^t * G_t) * d(log(pi))

    Args:
      rewards (np.ndarray): sequence of rewards observed by the agent
      discount_rate (float): `gamma`
    """
    discounted_rates = np.vstack([
        [[1.]],
        np.repeat([[discount_rate]], len(rewards) - 1, axis=0).cumprod(axis=0)])
    rewards_discounted = rewards * discounted_rates
    return np.flipud(np.flipud(rewards_discounted).cumsum(axis=0))

def collect_episode(env, policy, render=False):
    episode = []
    done = False
    obs = env.reset()
    while not done:
        if render:
            env.render()
        action = policy(obs)
        obs_next, reward, done, _ = env.step(action)
        episode.append(Experience(state=obs, action=action, reward=reward, done=done))
        obs = obs_next
    return episode


class ExponentialMovingAverageVariance:
    def __init__(self, shape, alpha):
        """Initializes

        Args:
          shape (tuple): the shape of the value
          alpha (float): represents the degree of weighting decrease,
            a constant smoothing factor between 0 and 1. A higher `alpha`
            discounts older observations faster.
        """
        self.mean = np.zeros(shape)
        self.variance = np.zeros(shape)
        self.alpha = alpha
        self.first = True

    def update(self, new_val):
        if self.first:
            np.copyto(self.mean, new_val)
            self.first = False
        else:
            delta = new_val - self.mean
            self.mean = self.mean + self.alpha * delta
            self.variance = (1 - self.alpha) * (self.variance + self.alpha * delta**2)

    def stddev(self):
        return np.sqrt(self.variance)
