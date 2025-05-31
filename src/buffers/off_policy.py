import numpy as np
import torch
from typing import Union, Tuple

class ReplayBuffer:
    """
    Off-policy RL 向け汎用リプレイバッファ。
    add に渡された各要素は内部で NumPy 配列化し、
    sample でまとめて Torch Tensor にして返却します。
    """
    def __init__(self,
                 buffer_size: int,
                 state_shape: Union[int, Tuple[int, ...]],
                 action_dim: int = 2,
                 device: str = "cpu"):
        """
        Args:
            buffer_size (int): バッファの最大サイズ
            state_shape (int or tuple): 状態ベクトルの形状 (例: 1080 または (1080,))
            action_dim (int): 行動ベクトルの次元
            device (str): サンプリング時に返す Tensor のデバイス
        """
        self.buffer_size = buffer_size
        self.device = device

        # state_shape が int ならタプル化
        if isinstance(state_shape, int):
            self.state_shape = (state_shape,)
        else:
            self.state_shape = state_shape
        self.action_dim = action_dim

        # バッファ初期化
        self.reset()

    def reset(self):
        """バッファを空にしてすべての配列を再確保"""
        self.pos = 0
        self.full = False
        # storage にすべてまとめる
        self.storage = {
            "states":      np.zeros((self.buffer_size, *self.state_shape), dtype=np.float32),
            "actions":     np.zeros((self.buffer_size, self.action_dim),    dtype=np.float32),
            "rewards":     np.zeros((self.buffer_size, 1),                  dtype=np.float32),
            "next_states": np.zeros((self.buffer_size, *self.state_shape),  dtype=np.float32),
            "dones":       np.zeros((self.buffer_size, 1),                  dtype=np.float32),
        }

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        """入力を float32 の np.ndarray に変換"""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)
        return x.astype(np.float32)

    def add(self, state, action, reward, next_state, done):
        """
        1 タイムステップ分のデータを追加。
        どの型で渡されても NumPy 配列として格納。
        """
        idx = self.pos
        # 各キーに対して _to_numpy → 代入
        self.storage["states"][idx]      = self._to_numpy(state)
        self.storage["actions"][idx]     = self._to_numpy(action)
        self.storage["rewards"][idx, 0]  = float(self._to_numpy(reward))
        self.storage["next_states"][idx] = self._to_numpy(next_state)
        self.storage["dones"][idx, 0]    = float(self._to_numpy(done))

        # ポインタ更新
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int):
        """
        ミニバッチをランダムサンプリングし，Tensor で返却。
        returns: states, actions, rewards, next_states, dones
        """
        max_i = self.buffer_size if self.full else self.pos
        idxs = np.random.choice(max_i, size=batch_size, replace=False)

        # 各配列を Torch Tensor に変換＆デバイス転送
        batch = {}
        for key, arr in self.storage.items():
            batch[key] = torch.from_numpy(arr[idxs]).to(self.device)

        return (
            batch["states"],
            batch["actions"],
            batch["rewards"],
            batch["next_states"],
            batch["dones"],
        )

    def __len__(self):
        return self.buffer_size if self.full else self.pos
