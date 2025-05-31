import numpy as np

class E2ETransform:
    def __init__(self, range_max=10.0, base_num=1081, downsample_num=181):
        self.range_max = range_max
        self.base_num = base_num
        self.downsample_num = downsample_num
        self.sample_indices = np.round(np.linspace(0, base_num - 1, downsample_num)).astype(int)

    def __call__(self, scan, steer, speed):
        # scan 正規化 + ダウンサンプリング
        scan = np.clip(scan, 0, self.range_max) / self.range_max
        scan = scan[self.sample_indices]

        # steer / speed のクリップ
        steer = np.clip(steer, -1.0, 1.0)
        speed = np.clip(speed, -1.0, 1.0)

        return scan, steer, speed
