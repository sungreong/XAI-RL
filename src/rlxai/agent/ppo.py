import torch
import torch.nn as nn
from torch.distributions import Categorical
from rlxai.agent.model import NeuralNetwork

# from rlxai.core.buffer.rolloutbuffer import  RolloutBuffer
from rlxai.buffer.rollout_buffer import RolloutBuffer
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = NeuralNetwork(state_dim, action_dim)

        # critic
        self.critic = NeuralNetwork(state_dim, 1)

    def __call__(self, state):
        action_logits = self.actor(state)
        state_values = self.critic(state)
        return action_logits, state_values

    def act(self, state):

        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        batch_size,
        eps_clip,
        device="cpu",
    ):
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.clip_grad_norm = 2.0
        self._lambda = 0.95
        self.vf_coef = 1.0
        self.ent_coef = 0.01
        self.buffer = RolloutBuffer()

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

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_next_states = torch.squeeze(torch.stack(self.buffer.next_states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        reward = torch.tensor(self.buffer.rewards).squeeze()
        done = torch.tensor(np.array(self.buffer.is_terminals, dtype=bool) * 1)
        with torch.no_grad():
            next_value = self.policy.critic(old_next_states[[-1], :]).squeeze().cpu()
            value = self.policy.critic(old_states).squeeze().cpu()
            delta = reward + (1 - done) * self.gamma * next_value - value
            adv = delta.clone()
            adv, done = adv.view(-1, len(reward)), done.view(-1, len(reward))
            for t in reversed(range(len(reward) - 1)):
                adv[:, t] += (1 - done[:, t]) * self.gamma * self._lambda * adv[:, t + 1]
            # if self.use_standardization:
            # adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True) + 1e-7)
            adv = adv.view(-1, 1).to(self.device)
            ret = adv + value.view(-1, 1).to(self.device)
        idxs = np.arange(len(old_states))
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for offset in range(0, len(old_states), self.batch_size):
                idx = idxs[offset : offset + self.batch_size]
                batch_old_state, batch_old_actions, batch_old_logprobs, batch_ret, batch_adv = map(
                    lambda x: x[idx], [old_states, old_actions, old_logprobs, ret, adv]
                )
                batch_logprobs, batch_state_values, batch_dist_entropy = self.policy.evaluate(
                    batch_old_state, batch_old_actions
                )
                batch_state_values = torch.squeeze(batch_state_values)
                ratios = torch.exp(batch_logprobs - batch_old_logprobs.detach())

                surr1 = ratios * batch_adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_adv

                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = self.MseLoss(batch_state_values, batch_ret).mean()

                entropy_loss = -batch_dist_entropy.mean()

                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
                self.optimizer.step()

            # # Evaluating old actions and values
            # logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # # match state_values tensor dimensions with rewards tensor
            # state_values = torch.squeeze(state_values)

            # # Finding the ratio (pi_theta / pi_theta__old)
            # ratios = torch.exp(logprobs - old_logprobs.detach())

            # # Finding Surrogate Loss
            # advantages = rewards - state_values.detach()
            # surr1 = ratios * advantages
            # surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # # final loss of clipped objective PPO
            # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # # take gradient step
            # self.optimizer.zero_grad()
            # loss.mean().backward()
            # self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
