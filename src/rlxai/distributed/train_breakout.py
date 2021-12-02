from rlxai.distributed.distributed_manager import DistributedManager

import os, sys, inspect, re, traceback
from collections import OrderedDict
import gym
from rlxai.agent.agent_selection import get_agent

env_config = dict(env_name="BreakoutDeterministic-v4")
model_name = "PPO"
device = "cpu"
save_path = "./model"
env = gym.make(env_config.get("env_name"))  # Deterministic

action_dim = env.action_space.n
stack_n = 4

agent_config = dict(
    model_name=model_name,
    save_path=save_path,
    device=device,
    action_dim=action_dim,
    state_dim=stack_n,
    load_model=True,
)

distributed_manager = DistributedManager(env_config, agent_config, num_workers=2, mode="sync")
agent = get_agent(**agent_config)
run_step = 1_000_000_000
step, print_stamp, save_stamp = 0, 0, 0
update_period = 3_000
while step < run_step:
    transitions = distributed_manager.run(update_period)
    step += update_period
    print_stamp += update_period
    save_stamp += update_period
    result = agent.process(transitions, step)
    distributed_manager.sync(agent.sync_out())

    if step >= run_step:
        agent.save(save_path)
        save_stamp = 0
