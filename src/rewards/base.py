from abc import ABC

class RewardBase(ABC):
    def __init__(self, ratio: float=1.0):
        self.ratio = ratio

    def get_reward(self, obs, pre_obs, action=None):
        ## 1 lap
        if obs["lap_counts"][0] == 2:
            return 1.0
        ## collisions
        if obs["collisions"][0]:
            return -1.0
        ## spin
        if obs["poses_theta"][0] > 100.0:
            return -1.0
        
        return 0.0

    def reset(self):
        pass
