import torch
import torch.nn as nn
from torch.distributions import Categorical
from rlxai.agent.model import NeuralNetwork
import numpy as np
from rlxai.buffer.rollout_buffer import RolloutBuffer
from torch.nn import functional as F
from torchvision import models


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # model = models.resnet18(pretrained=True)
        # model.fc = nn.Linear(512, action_dim)
        self.actor = NeuralNetwork(state_dim, action_dim)
        # model = models.resnet18(pretrained=True)
        # model.fc = nn.Linear(512, 1)
        # critic
        self.critic = NeuralNetwork(state_dim, 1)

    def act(self, state, training=False):

        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        if training:
            action = dist.sample()
        else:
            action = torch.argmax(action_logits, keepdim=True).unsqueeze(dim=0)

        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def __call__(self, state):
        action_logits = self.actor(state)
        v = self.critic(state)
        pi = F.softmax(action_logits, dim=-1)
        return pi, v

    def evaluate(self, state, action):

        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


# if torch.isnan(x).sum() > 0:
#             raise Exception("step_x")
from .base import BaseAgent


class PPO(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        batch_size,
        eps_clip=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        lambda_=0.95,
        clip_grad_norm=0.3,
        device="cpu",
        n_step=128,
    ):
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.epsilon_clip = eps_clip
        self.K_epochs = K_epochs
        self.clip_grad_norm = clip_grad_norm
        self._lambda = lambda_
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.time_t = 0
        self.learn_stamp = 0
        self.n_step = n_step

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.memory = RolloutBuffer()

    def evluate(self, state):
        with torch.no_grad():
            self.policy_old

    def select_action(self, state, training=False):
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(self.as_tensor(state), training)
        return {"action": action.cpu().numpy()[np.newaxis, :]}

    def update(self):
        ## jorldy code
        transitions = self.memory.sample()
        from collections import Counter

        print(Counter(transitions["action"].squeeze().tolist()))
        for key in transitions.keys():
            transitions[key] = self.as_tensor(transitions[key])
        state = transitions["state"]
        action = transitions["action"]
        reward = transitions["reward"]
        next_state = transitions["next_state"]
        done = transitions["done"]
        print(action.size(), reward.sum(), done.sum())
        # convert list to tensor
        # set prob_a_old and advantage
        with torch.no_grad():
            pi, value = self.policy_old(state)
            prob = pi.gather(1, action.long())
            prob_old = prob

            next_value = self.policy_old(next_state)[-1]
            delta = reward + (1 - done) * self.gamma * next_value - value
            adv = delta.clone()
            adv, done = adv.view(-1, self.n_step), done.view(-1, self.n_step)
            for t in reversed(range(self.n_step - 1)):
                adv[:, t] += (1 - done[:, t]) * self.gamma * self._lambda * adv[:, t + 1]
            # if self.use_standardization:
            # adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True) + 1e-7)
            adv = adv.view(-1, 1)
            ret = adv + value

        # start train iteration
        actor_losses, critic_losses, entropy_losses, ratios, probs = [], [], [], [], []
        idxs = np.arange(len(reward))
        for _ in range(self.K_epochs):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), self.batch_size):
                idx = idxs[offset : offset + self.batch_size]

                _state, _action, _ret, _next_state, _adv, _prob_old = map(
                    lambda x: [_x[idx] for _x in x] if isinstance(x, list) else x[idx],
                    [state, action, ret, next_state, adv, prob_old],
                )
                pi, value = self.policy(_state)
                m = Categorical(pi)
                prob = pi.gather(1, _action.long())

                ratio = (prob / (_prob_old + 1e-7)).prod(1, keepdim=True)
                surr1 = ratio * _adv
                surr2 = torch.clamp(ratio, min=1 - self.epsilon_clip, max=1 + self.epsilon_clip) * _adv
                actor_loss = -torch.min(surr1, surr2).sum()

                # critic_loss = F.mse_loss(value, _ret).mean()
                critic_loss = F.smooth_l1_loss(value, _ret).mean()
                entropy_loss = -m.entropy().mean()

                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                probs.append(prob.min().item())
                ratios.append(ratio.max().item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
        result = {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy_loss": np.mean(entropy_losses),
            "max_ratio": max(ratios),
            "min_prob": min(probs),
            "min_prob_old": prob_old.min().item(),
        }
        print(result)
        self.policy_old.load_state_dict(self.policy.state_dict())
        return result

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.learn_stamp += delta_t
        # Process per epi
        if self.learn_stamp >= self.n_step:
            result = self.update()
            self.learn_stamp = 0

        return result

    def sync_in(self, weights):
        self.policy_old.load_state_dict(weights)

    def sync_out(self, device="cpu"):
        weights = self.policy.state_dict()
        for k, v in weights.items():
            weights[k] = v.to(device)
        sync_item = {
            "weights": weights,
        }
        return sync_item
