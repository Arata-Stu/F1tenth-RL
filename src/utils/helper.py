import numpy as np
from collections import deque
import torch

def soft_update(source, target, tau=0.005):
    """
    Soft Update for target network parameters
    Args:
        source (nn.Module): メインのネットワーク
        target (nn.Module): ターゲットネットワーク
        tau (float): ソフトアップデートの係数
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)



def convert_action(action, steer_range: float = 0.4, speed_range: float = 1.0):
    """
    生の行動値（ステアリングとスロットル）を、スケール調整されたステアリングとスピードに変換します。
    この関数は、torch.Tensor、リスト、または np.ndarray 型の入力を処理できます。

    ステアリング計算: output = raw_steer * steer_range
    スピード計算:   output = (raw_throttle + 1.0) / 2.0 * speed_range
                   スピードは speed_range の値で上限が設定されます。

    speed_range のデフォルト値に関する注意:
    元の `convert_action` 関数の speed_range のデフォルト値は 10.0 でしたが、
    `convert_action_torch` 関数では 1.0 でした。この統一関数では、
    `convert_action_torch` に合わせて 1.0 をデフォルト値として使用します。
    必要に応じて呼び出し時に値を調整してください。

    Args:
        action: 生の行動値。2つの要素（ステアリング、スロットル）を持つシーケンス、
                またはそのようなシーケンスのバッチを想定しています（例: shape (2,) または (N, 2)）。
                生のステアリング値とスロットル値は、通常 [-1.0, 1.0] の範囲内であると想定されます。
        steer_range (float): ステアリングの最大絶対値。デフォルトは 0.4 です。
        speed_range (float): スピードの最大値。デフォルトは 1.0 です。

    Returns:
        変換された行動値 [steer, speed] またはそのバッチ。
        入力型（torch.Tensor, list, または np.ndarray）に一致する型で返されます。
    """
    is_tensor = isinstance(action, torch.Tensor)
    is_list = isinstance(action, list)

    if is_tensor:
        # PyTorch Tensor の処理
        _steer_range_t = torch.tensor(steer_range, device=action.device, dtype=action.dtype)
        _speed_range_t = torch.tensor(speed_range, device=action.device, dtype=action.dtype)

        # action の形状が (..., 2) であると仮定
        raw_steer = action[..., 0]
        raw_throttle = action[..., 1]

        steer = raw_steer * _steer_range_t
        
        speed_normalized = (raw_throttle + 1.0) / 2.0
        speed_scaled = speed_normalized * _speed_range_t
        speed = torch.minimum(speed_scaled, _speed_range_t)
        
        output_action = torch.stack((steer, speed), dim=-1)
        return output_action
        
    elif is_list or isinstance(action, np.ndarray):
        # リストまたは NumPy 配列の処理
        # リストの場合、np.ndarray に変換。NumPy 配列の場合は float32 型に変換。
        action_np = np.array(action, dtype=np.float32) if is_list else action.astype(np.float32)

        if action_np.shape[-1] != 2:
            raise ValueError(
                f"action の最後の次元は2（ステアリング、スロットル）である必要がありますが、"
                f"形状 {action_np.shape} を受け取りました。"
            )

        # action_np の形状が (..., 2) であると仮定
        raw_steer = action_np[..., 0]
        raw_throttle = action_np[..., 1]

        steer = raw_steer * steer_range
        
        speed_normalized = (raw_throttle + 1.0) / 2.0
        speed_scaled = speed_normalized * speed_range
        speed = np.minimum(speed_scaled, speed_range)

        output_action_np = np.stack((steer, speed), axis=-1).astype(np.float32)
        
        if is_list:
            return output_action_np.tolist()  # 入力がリストならリストで返す
        else:
            return output_action_np  # 入力がNumPy配列ならNumPy配列で返す
            
    else:
        raise TypeError(
            f"サポートされていない action の型です: {type(action)}。 "
            "torch.Tensor, list, または np.ndarray を想定しています。"
        )

def convert_scan(scans, scan_range: float=30.0):
    scans = scans / scan_range
    scans = np.clip(scans, 0, 1)
    return scans


class ScanBuffer:
    def __init__(self, frame_size: int = 1080,
                 num_scan: int = 2,
                 target_size: int = 60):
        """
        frame_size: number of points per raw scan (e.g., 1080)
        num_scan: number of frames to buffer/concatenate
        target_size: if specified, downsample each scan to this length by equal-interval sampling
        """
        self.frame_size = frame_size
        self.num_scan = num_scan
        self.target_size = target_size
        self.scan_window = deque(maxlen=num_scan)

    def add_scan(self, scan: np.ndarray):
        """Add a new scan; must be length frame_size."""
        if scan.shape[0] != self.frame_size:
            raise ValueError(f"scan length {scan.shape[0]} != expected {self.frame_size}")
        self.scan_window.append(scan)

    def is_full(self) -> bool:
        """Check if buffer has num_scan frames."""
        return len(self.scan_window) == self.num_scan
    
    def reset(self):
        """Clear the scan buffer."""
        self.scan_window.clear()

    def _pad_frames(self, frames: list) -> list:
        """
        If fewer than num_scan frames, repeat the last frame to pad up to num_scan.
        """
        if not frames:
            raise ValueError("No frames in buffer")
        if len(frames) < self.num_scan:
            last = frames[-1]
            frames = frames + [last] * (self.num_scan - len(frames))
        return frames

    def _downsample(self, arr: np.ndarray) -> np.ndarray:
        """
        Downsample a 1D array to target_size points by equal-interval sampling.
        """
        if self.target_size is None or arr.size == self.target_size:
            return arr
        indices = np.linspace(0, arr.size - 1, self.target_size, dtype=int)
        return arr[indices]

    def get_concatenated_numpy(self) -> np.ndarray:
        """
        Return concatenated frames as a NumPy array, downsampling by equal-interval if target_size is set.
        """
        frames = list(self.scan_window)
        frames = self._pad_frames(frames)
        processed = [self._downsample(f) for f in frames]
        return np.hstack(processed)

    def get_concatenated_tensor(self,
                                device: torch.device = None,
                                dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Return concatenated frames as a PyTorch tensor, downsampling by equal-interval if target_size is set.
        """
        frames = list(self.scan_window)
        frames = self._pad_frames(frames)
        tensors = []
        for f in frames:
            arr = self._downsample(f) if isinstance(f, np.ndarray) else f.numpy()
            t = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else f
            tensors.append(t)
        out = torch.cat(tensors, dim=0)
        if device:
            out = out.to(device)
        if dtype:
            out = out.to(dtype)
        return out
