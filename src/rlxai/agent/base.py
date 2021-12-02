import torch
from abc import abstractmethod


class BaseAgent(object):
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
            if torch.sum(torch.isnan(x)) > 0:
                raise Exception(f"tensor have a nan : {x}")
        return x

    @abstractmethod
    def save(self, path):
        """
        Save model to path.
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load model from path.
        """
        pass

    def sync_in(self, weights):
        self.network.load_state_dict(weights)

    def sync_out(self, device="cpu"):
        weights = self.network.state_dict()
        for k, v in weights.items():
            weights[k] = v.to(device)
        sync_item = {
            "weights": weights,
        }
        return sync_item

    def set_distributed(self, *args, **kwargs):
        return self

    def interact_callback(self, transition):
        return transition
