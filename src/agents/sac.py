import torch
import torch.nn as nn
import torch.optim as optim

from src.models.actor import get_actor
from src.models.critic import get_critic
from src.utils.helper import soft_update

class SAC:
    def __init__(self,
                 actor_cfg,
                 critic_cfg,
                 alpha_lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 target_entropy=None,
                 device="cpu"):
        self.device = device
        
        # Actor, Critic の初期化
        self.actor = get_actor(state_dim=actor_cfg.state_dim,
                               action_dim=actor_cfg.action_dim,
                               hidden_dim=actor_cfg.hidden_dim,
                               policy_type=actor_cfg.name).to(device)

        self.critic = get_critic(state_dim=critic_cfg.state_dim,
                               action_dim=critic_cfg.action_dim,
                               hidden_dim=critic_cfg.hidden_dim,
                               type=critic_cfg.name).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_cfg.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_cfg.lr)

        # 温度パラメータ α
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # ハイパーパラメータ
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy if target_entropy else -actor_cfg.action_dim

    def select_action(self, state, evaluate=False):
        """アクションの選択"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
                log_prob = None
            else:
                action, log_prob = self.actor.sample(state)
        return {
            "action": action.cpu().numpy()[0],
            "log_prob": None
        }

    def update(self, replay_buffer, batch_size=256):
        """ネットワークの更新処理"""
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # Critic の更新
        c_info = self._update_critic(state, action, reward, next_state, done)
        # Actor の更新
        a_info = self._update_actor(state)
        # Alpha (温度パラメータ) の更新
        alpha_info = self._update_alpha(state)
        
        # Soft Update を追加
        soft_update(self.critic.critic1, self.critic.target1, self.tau)
        soft_update(self.critic.critic2, self.critic.target2, self.tau)

        return {**c_info, **a_info, **alpha_info}

    def _update_critic(self, state, action, reward, next_state, done):
        """Critic の更新処理"""
        with torch.no_grad():
            na, nlp = self.actor.sample(next_state)
            alpha = self.log_alpha.exp()
            tq1, tq2 = self.critic.target(next_state, na)
            q_target = torch.min(tq1, tq2) - alpha * nlp
            target_q = reward + (1 - done) * self.gamma * q_target

        # 現在の Q 値を取得
        q1, q2 = self.critic.forward(state, action)
        loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

        # オプティマイザの更新
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        return {'critic_loss': loss.item()}

    def _update_actor(self, state):
        """Actor の更新処理"""
        na, lp = self.actor.sample(state)
        q1, q2 = self.critic.forward(state, na)
        q_min = torch.min(q1, q2)
        alpha = self.log_alpha.exp()
        loss = (alpha * lp - q_min).mean()

        # オプティマイザの更新
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        return {'actor_loss': loss.item()}

    def _update_alpha(self, state):
        """Alpha の更新処理"""
        _, lp = self.actor.sample(state)
        loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()

        # オプティマイザの更新
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()
        
        return {'alpha_loss': loss.item(), 'alpha_value': self.log_alpha.exp().item()}

    def save(self, path):
        """モデルの保存"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic.critic1.state_dict(),
            'critic2': self.critic.critic2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'alpha_opt': self.alpha_optimizer.state_dict()
        }, path)

    def load(self, path):
        """モデルのロード"""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.critic1.load_state_dict(ckpt['critic1'])
        self.critic.critic2.load_state_dict(ckpt['critic2'])
        self.critic.target1.load_state_dict(self.critic.critic1.state_dict())
        self.critic.target2.load_state_dict(self.critic.critic2.state_dict())
        self.log_alpha = ckpt['log_alpha']
        self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
        self.alpha_optimizer.load_state_dict(ckpt['alpha_opt'])
