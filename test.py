from gym import envs

print(envs.registry.all())


import gym

env = gym.make("Breakout-v0")
env.reset(mode="rgb_array")
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
env.close()
