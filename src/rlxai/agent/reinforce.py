import torch

torch.backends.cudnn.benchmark = True
from torch.distributions import Normal
import numpy as np
import os

from rlxai.agent.model import NeuralNetwork, DiscretePolicyValue
from rlxai.buffer.rollout_buffer import RolloutBuffer
from .base import BaseAgent


class REINFORCE(BaseAgent):
    """REINFORCE agent.
    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        hidden_size (int): dimension of hidden unit.
        network (str): key of network class in _network_dict.txt.
        head (str): key of head in _head_dict.txt.
        optim_config (dict): dictionary of the optimizer info.
        gamma (float): discount factor.
        use_standardization (bool): parameter that determine whether to use standardization for return.
        device (str): device to use.
            (e.g. 'cpu' or 'gpu'. None can also be used, and in this case, the cpu is used.)
    """

    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        use_standardization=False,
        device=None,
        **kwargs,
    ):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # assert self.action_type in ["continuous", "discrete"]

        self.network = DiscretePolicyValue(state_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=0.0003)

        self.gamma = gamma
        self.use_standardization = use_standardization
        self.memory = RolloutBuffer()

    @torch.no_grad()
    def select_action(self, state, training=True):
        self.network.train(training)
        # if self.action_type == "continuous":
        #     mu, std = self.network(self.as_tensor(state))
        #     z = torch.normal(mu, std) if training else mu
        #     action = torch.tanh(z)
        # else:
        pi = self.network(self.as_tensor(state))
        action = torch.multinomial(pi, 1) if training else torch.argmax(pi, dim=-1, keepdim=True)
        return {"action": action.cpu().numpy()}

    def learn(self):
        transitions = self.memory.sample()

        state = transitions["state"]
        action = transitions["action"]
        reward = transitions["reward"]

        ret = np.copy(reward)
        for t in reversed(range(len(ret) - 1)):
            ret[t] += self.gamma * ret[t + 1]
        if self.use_standardization:
            ret = (ret - ret.mean()) / (ret.std() + 1e-7)

        state, action, ret = map(lambda x: self.as_tensor(x), [state, action, ret])

        if self.action_type == "continuous":
            mu, std = self.network(state)
            m = Normal(mu, std)
            z = torch.atanh(torch.clamp(action, -1 + 1e-7, 1 - 1e-7))
            log_prob = m.log_prob(z)
        else:
            pi = self.network(state)
            log_prob = torch.log(pi.gather(1, action.long()))
        loss = -(log_prob * ret).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        result = {"loss": loss.item()}
        return result

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)

        # Process per epi
        if transitions[0]["done"]:
            result = self.learn()

        return result

    def save(self, checkpoint_path):
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
