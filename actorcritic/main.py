import tensorflow as tf
import gym

from .agent import Agent


def print_status(status):
    print(status)


def train(agent, make_env,
          num_iterations, max_time_steps, record_progress=print_status):
    """One-step actor-critic algorithm"""
    env: gym.Env = make_env()
    for iter_i in num_iterations:
        state = env.reset()
        for step_i in max_time_steps:
            action = agent.actor(state).sample()[0]
            next_state, reward, done, _ = env.step(action)
            status = agent.learn(state, action, reward, next_state, done)
            status = {'iteration': iter_i, 'step': step_i, **status}
            record_progress(status)
            if done:
                break


def main():
    def make_env():
        return gym.make('CartPole-v0')

    with tf.Session():
        env = make_env()
        agent = Agent(env.observation_space.shape, env.action_space.shape, discount_rate=0.99)
        train(agent, make_env, num_iterations=10, max_time_steps=1000)


if __name__ == "__main__":
    main()
