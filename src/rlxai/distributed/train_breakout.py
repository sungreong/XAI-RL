from rlxai.distributed.distributed_manager import DistributedManager

import os, sys, inspect, re, traceback
import matplotlib.pyplot as plt
from collections import OrderedDict
import gym
from rlxai.agent.agent_selection import get_agent
from rlxai.gym_wrapper import FireResetEnv, WarpFrame, ScaledFloatFrame, FrameStack

stack_n = 4
env_config = dict(env_name="Breakout-v4", stack_n=stack_n)  # Deterministic
# env_config = dict(env_name="BreakoutDeterministic-v4", stack_n=stack_n)  # Deterministic
# env_config = dict(env_name="BreakoutNoFrameskip-v4", stack_n=stack_n)  # Deterministic
model_name = "PPO"
device = "cpu"
save_path = "./model"
###############################################################
env = gym.make(env_config.get("env_name"))  # Deterministic
if "FIRE" in env.unwrapped.get_action_meanings():
    env = FireResetEnv(env)
env = WarpFrame(env)
env = ScaledFloatFrame(env)
env = FrameStack(env, k=stack_n)
env.seed(54852)
###############################################################
action_dim = env.action_space.n
agent_config = dict(
    model_name=model_name,
    save_path=save_path,
    device=device,
    action_dim=action_dim,
    state_dim=stack_n,
    load_model=True,
)
num_workers = 4
distributed_manager = DistributedManager(env_config, agent_config, num_workers=num_workers, mode="sync")
agent, model_algo_path, checkpoint_path = get_agent(**agent_config)
run_step = 1_000_000_000
step, print_stamp, save_stamp = 0, 0, 0
update_period = 100
BEST_REWARD = 0
REWARD_LIST = []
ACTION_LIST = []
from collections import Counter

ACTION_MEANING = env.unwrapped.get_action_meanings()

while step < run_step:
    print(f"{step}/{ run_step }")
    ################################
    transitions = distributed_manager.run(update_period)
    step += update_period
    result = agent.process(transitions, step)
    distributed_manager.sync(agent.sync_out())
    agent.save(checkpoint_path)
    ################################

    if step % (agent.n_step) == 0:
        state = env.reset()
        info = {"lives": 5}
        last_lives = info["lives"]
        SUM_OF_REWARDS = 0
        ACTION_LIST = []
        while True:
            action_dict = agent.select_action(state, False)
            if last_lives > info["lives"]:
                action = 1  # TODO: 에이전트가 1을 선택할 수 있게 하기?
                last_lives = info["lives"]
            else:
                action = int(action_dict["action"])
            state, reward, done, info = env.step(action)
            ACTION_LIST.append(ACTION_MEANING[action])
            SUM_OF_REWARDS += reward
            if done:
                break
        REWARD_LIST.append(SUM_OF_REWARDS)
        if BEST_REWARD < SUM_OF_REWARDS:
            BEST_REWARD = SUM_OF_REWARDS
            agent.save(checkpoint_path)
        print(Counter(ACTION_LIST))
        print(SUM_OF_REWARDS)
        plt.plot(REWARD_LIST)
        plt.title(f"SAVE MODEL Iter : {step}, MAX REWARD : {BEST_REWARD}")
        plt.savefig(model_algo_path.joinpath("./reward_graph.png"))
        plt.close()
