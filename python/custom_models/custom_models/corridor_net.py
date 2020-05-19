import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CorridorNet(nn.Module, TorchModelV2):
    def __init__(self, obs_space, action_space, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, *args, **kwargs)
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(obs_space.n * 2 - 1, 128)
        self.action_head = nn.Linear(128, action_space.n)
        self.critic_head = nn.Linear(128, 1)

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = F.relu(self.fc1(input_dict['obs']))
        action = self.action_head(x)
        value = self.critic_head(x)
        self._value_out = value.reshape([-1])
        return action, state

    def value_function(self):
        return self._value_out
