import copy
import torch
import torch.nn as nn

from .backbone import TinyLidarBackbone

def get_critic(state_dim, action_dim, hidden_dim, type='mlp'):
    """
    Criticネットワークのインスタンスを取得
    """
    if type == 'mlp':
        return DoubleCritic(state_dim, action_dim, hidden_dim)
    elif type == '1dconv':
        return Double1dConvCritic(state_dim, action_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown critic type: {type}")

class DoubleCritic(nn.Module):
    """
    二重Criticネットワークとターゲットの管理
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        # Q1, Q2 ネットワーク
        def make_net():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        self.critic1 = make_net()
        self.critic2 = make_net()
        # ターゲットネットワーク
        self.target1 = copy.deepcopy(self.critic1)
        self.target2 = copy.deepcopy(self.critic2)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.critic1(x)
        q2 = self.critic2(x)
        return q1, q2

    def target(self, state, action):
        x = torch.cat([state, action], dim=1)
        tq1 = self.target1(x)
        tq2 = self.target2(x)
        return tq1, tq2


class Double1dConvCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        
        self.lidar_backbone = TinyLidarBackbone(input_dim=state_dim)
        def make_net():
            return nn.Sequential(
                nn.Linear(self.lidar_backbone.out_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        self.critic1 = make_net()
        self.critic2 = make_net()
        self.target1 = copy.deepcopy(self.critic1)
        self.target2 = copy.deepcopy(self.critic2)

    def forward(self, state, action):
        features = self.lidar_backbone(state)
        x = torch.cat([features, action], dim=1)
        q1 = self.critic1(x)
        q2 = self.critic2(x)
        return q1, q2

    def target(self, state, action):
        features = self.lidar_backbone(state)
        x = torch.cat([features, action], dim=1)
        tq1 = self.target1(x)
        tq2 = self.target2(x)
        return tq1, tq2