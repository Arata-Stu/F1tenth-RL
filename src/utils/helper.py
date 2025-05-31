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



def convert_action(action, steer_range: float=0.4, speed_range: float=10.0):
    
    steer = action[0] * steer_range
    speed = (action[1] + 1) / 2 * speed_range
    speed = min(speed, speed_range)
    action = [steer, speed]
    return action

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
