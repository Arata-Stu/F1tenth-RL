import numpy as np
from .base import RewardBase
from f1tenth_gym.maps.map_manager import MapManager

class ProgressReward(RewardBase):
    def __init__(self, ratio: float=1.0, map_manager: MapManager=None):
        super().__init__()
        self.ratio = ratio
        self.map_manager = map_manager

    def get_reward(self, obs, pre_obs, action=None):

        base_reward = super().get_reward(obs, pre_obs)
        

        current_position = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        prev_position = np.array([pre_obs['poses_x'][0], pre_obs['poses_y'][0]])
        
        current_progress = self.map_manager.calc_progress(current_position)
        prev_progress = self.map_manager.calc_progress(prev_position)
        distance = current_progress - prev_progress ## 例 0.007
        
        ## 大きな値は無視
        if abs(distance) > 1.0:
            distance = 0.0

        ## max 1.0
        progress_reward = distance * 50.0
        progress_reward *= self.ratio
        
        return base_reward + progress_reward
    
    def reset(self):
        pass