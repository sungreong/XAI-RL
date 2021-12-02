from rlxai.agent.model import NeuralNetwork
from torch import nn
import torch
import numpy as np


class ReinforceGA(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(ReinforceGA, self).__init__()

        self.actor = NeuralNetwork(state_dim, action_dim).to(device)
        self.device = device

    def select_action(self, state, training=False):
        action_logits = self.actor(self.as_tensor(state))
        action = torch.argmax(action_logits, keepdim=True).unsqueeze(dim=0)
        return {"action": action.cpu().numpy()[np.newaxis, :]}

    def as_tensor(self, x):
        if isinstance(x, list):
            x = list(
                map(
                    lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device),
                    x,
                )
            )
        else:
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return x

    def save(self, path):
        # print(f"...Save model to {path}...")
        torch.save(
            {
                "network": self.actor.state_dict(),
            },
            path,
        )

    def load(self, path):
        # print(f"...Load model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["network"])
