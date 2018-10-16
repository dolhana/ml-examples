"""Never Ending Pendulum with Random Agent"""
import gym

env = gym.make('Pendulum-v0')
_s0 = env.reset()
done = False
env.render()

for i_step in range(10**10):
    action = env.action_space.sample()
    _s1, _r, done, _info = env.step(action)
    _s0 = _s1
    env.render()
    if (i_step % 100) == 0:
        print(i_step, _s0, _r, done)
    if done:
        env.reset()

env.close()
