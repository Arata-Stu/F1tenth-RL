import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from src.agents.sac import SAC
from src.models.actor import get_actor

def get_agent(agent_cfg: DictConfig, device: str):
    """
    エージェントの初期化
    """
    agent_name = agent_cfg.name
    if agent_name == "sac":
        return SAC(
            actor_cfg=agent_cfg.actor,
            critic_cfg=agent_cfg.critic,
            alpha_lr=agent_cfg.alpha_lr,
            gamma=agent_cfg.gamma,
            tau=agent_cfg.tau,
            target_entropy=agent_cfg.target_entropy,
            device=device
        )
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")
    

class TrainAgent:
    def __init__(self, actor_cfg, device="cpu", lr=1e-3):
        self.device = device
        self.actor = get_actor(actor_cfg=actor_cfg).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)

    def predict(self, state):
        """
        状態（単体 or バッチ）に対してアクションを推論する
        入力: state.shape = (B, state_dim) または (state_dim,)
        出力: action.shape = (B, action_dim) または (action_dim,)
        """
        state = torch.FloatTensor(state).to(self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            mean, _ = self.actor(state)
            action = torch.tanh(mean)
        return action.cpu().numpy() if action.shape[0] > 1 else action.cpu().numpy()[0]

    def train_step(self, state, target_action):
        """
        教師あり学習の1ステップ（バッチ対応）
        state.shape = (B, state_dim)
        target_action.shape = (B, action_dim)
        """
        self.actor.train()

        state = torch.FloatTensor(state).to(self.device)
        target = torch.FloatTensor(target_action).to(self.device)

        mean, _ = self.actor(state)
        pred_action = torch.tanh(mean)

        loss = self.loss_fn(pred_action, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])

    def save(self, path):
        torch.save({'actor': self.actor.state_dict()}, path)


class TestAgent:
    def __init__(self, actor_cfg, device="cpu"):
        self.device = device
        self.actor = get_actor(actor_cfg=actor_cfg).to(self.device)
        self.actor.eval()  # 推論モードに設定

    def select_action(self, state, mode="test"):
        """
        状態に対してアクションを選択する
        mode: "test"（決定論的） / "train"（確率的）
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if mode == "test":
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            elif mode == "train":
                action, _ = self.actor.sample(state)
            else:
                raise ValueError(f"Unknown mode '{mode}'. Use 'test' or 'train'.")
        return action.cpu().numpy()[0]

    def load(self, path):
        """
        学習済みのactorモデルの読み込み
        """
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
