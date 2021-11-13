from torch.optim import Adam
from torch.distributions import Categorical
import torch
import numpy as np
import torch.nn.functional as F
import scipy.signal
import os


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class A2C(object):
    def __init__(
        self,
        actor,
        critic,
        lr=1e-3,
        gamma=0.9,
        lam=0.1,
        device="cpu",
    ):
        self._optimizerA = Adam(
            actor.parameters(),
            lr=lr,
        )
        self._optimizerC = Adam(
            critic.parameters(),
            lr=lr,
        )
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lam = lam
        self.device = device

    def __str__(
        self,
    ):
        return "A2C"

    def choose_action(self, state, inference=False):
        logits = self.actor(state)
        value = self.critic(state)
        distribution = Categorical(logits=logits)
        if inference:
            model_action = torch.argmax(distribution.probs).detach().cpu().numpy()
            return model_action, value
        else:
            model_action = distribution.sample()
            logprob = distribution.log_prob(model_action)
            return model_action, value, logprob

    def finish_trajectory(self, rewards, is_terminals, values, last_value=0, gamma=0.9, lam=0.1):

        start_idx = 0
        r = rewards[start_idx:]
        d = is_terminals[start_idx:]
        v = values[start_idx:]
        v = torch.cat(v).squeeze().detach().cpu().numpy()
        r = np.append(r, last_value)
        v = np.append(v, last_value)
        deltas = r[:-1] + gamma * v[1:] - v[:-1]
        advs_arr = discounted_cumulative_sums(deltas, gamma * lam)
        returns_arr = discounted_cumulative_sums(r, gamma)[:-1]
        advs = np.zeros(len(advs_arr))
        returns = np.zeros(len(advs_arr))
        advs[start_idx:] = advs_arr
        returns[start_idx:] = returns_arr
        return advs, returns

    def train(self, actor_loss, critic_loss):
        self._optimizerA.zero_grad()
        self._optimizerC.zero_grad()
        actor_loss.backward(retain_graph=True)
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 2)
        self._optimizerA.step()
        self._optimizerC.step()

    def get_loss(self, rewards, is_terminals, values, logprobs, last_value):
        advs, returns = self.finish_trajectory(rewards, is_terminals, values, last_value, self.gamma, self.lam)

        values = torch.stack(values).squeeze()
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device).detach()
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device).detach()
        logprobs = torch.stack(logprobs).squeeze()
        actor_loss, critic_loss = self.a2c_losses(values, logprobs, returns, advs)
        return actor_loss, critic_loss

    def a2c_losses(self, values, logprobs, rs, advs):
        r = rs.view(-1, 1)
        advs = advs.view(-1, 1)
        logprobs = logprobs.view(-1, 1)  # advantage = (r_batch - values).detach()
        # logprobs = torch.clamp(logprobs, min=-10, max=10)
        v = values.view(-1, 1)
        actor_loss = (-logprobs * advs).mean()
        critic_loss = F.smooth_l1_loss(v, r).mean()  # , reduction="sum"
        return actor_loss, critic_loss

    def save_model(self, folder):
        PATH = os.path.join(folder, "model.pt")
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, PATH)

    def load_model(self, folder):
        PATH = os.path.join(folder, "model.pt")
        model_dict = torch.load(PATH)
        self.actor.load_state_dict(model_dict["actor"])
        self.critic.load_state_dict(model_dict["critic"])
