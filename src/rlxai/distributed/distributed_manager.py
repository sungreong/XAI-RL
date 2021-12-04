import os
from functools import reduce

import ray
import numpy as np
from rlxai.agent.agent_selection import get_agent
from rlxai.gym_wrapper import FireResetEnv, WarpFrame, ScaledFloatFrame, FrameStack
import gym


class DistributedManager:
    def __init__(self, env_config, agent_config, num_workers, mode):
        try:
            ray.init(address="auto")
        except:
            ray.init()
        agent, _, _ = get_agent(**agent_config)
        self.num_workers = num_workers if num_workers else os.cpu_count()
        env_config, agent = map(ray.put, [env_config, agent])
        self.actors = [Actor.remote(env_config, agent, (i + 1) * 10) for i in range(self.num_workers)]

        assert mode in ["sync", "async"]
        self.mode = mode
        self.sync_item = None
        self.running_ids = []

    def run(self, step=1):
        assert step > 0
        if self.mode == "sync":
            items = ray.get([actor.run.remote(step) for actor in self.actors])
            transitions = reduce(lambda x, y: x + y, [item[1] for item in items])
        else:
            if len(self.running_ids) == 0:
                self.running_ids = [actor.run.remote(step) for actor in self.actors]

            done_ids = []
            while len(done_ids) == 0:
                done_ids, self.running_ids = ray.wait(self.running_ids, num_returns=self.num_workers, timeout=0.1)

            items = ray.get(done_ids)
            transitions = reduce(lambda x, y: x + y, [item[1] for item in items])
            runned_ids = [item[0] for item in items]

            if self.sync_item is not None:
                ray.get([self.actors[id].sync.remote(self.sync_item) for id in runned_ids])
            self.running_ids += [self.actors[id].run.remote(step) for id in runned_ids]

        return transitions

    def sync(self, sync_item):
        if self.mode == "sync":
            sync_item = ray.put(sync_item)
            ray.get([actor.sync.remote(sync_item) for actor in self.actors])
        else:
            self.sync_item = ray.put(sync_item)

    def terminate(self):
        if len(self.running_ids) > 0:
            ray.get(self.running_ids)
        ray.shutdown()


@ray.remote
class Actor:
    def __init__(self, env_config, agent, id):
        self.id = id
        ##############################################
        self.env = gym.make(env_config.get("env_name"))
        if "FIRE" in self.env.unwrapped.get_action_meanings():
            self.env = FireResetEnv(self.env)
        self.env = WarpFrame(self.env)
        self.env = ScaledFloatFrame(self.env)
        self.env = FrameStack(self.env, k=env_config.get("stack_n"))
        ##########################################
        self.agent = agent.set_distributed(id)
        self.env.seed(id)
        self.state = self.env.reset()
        # self.info = {"lives": 5}

    def run(self, step):
        transitions = []
        # last_lives = self.info["lives"]
        for t in range(step):
            action_dict = self.agent.select_action(self.state, training=True)
            # if last_lives > self.info["lives"]:
            #     action = 1  # TODO: 에이전트가 1을 선택할 수 있게 하기?
            #     last_lives = self.info["lives"]
            # else:
            #     action = int(action_dict["action"])
            action = int(action_dict["action"])
            next_state, reward, done, self.info = self.env.step(action)
            reward, done = map(lambda x: np.expand_dims(x, 0), [[reward], [done]])  # for (1, ?)
            transition = {
                "state": self.state,
                "next_state": next_state,
                "reward": reward,
                "done": done,
            }
            transition.update(action_dict)
            transition = self.agent.interact_callback(transition)
            if transition:
                transitions.append(transition)
            self.state = next_state if not done else self.env.reset()
            # if done:
            #     self.info = {"lives": 5}
            #     last_lives = self.info["lives"]

        return self.id, transitions

    def sync(self, sync_item):
        self.agent.sync_in(**sync_item)
