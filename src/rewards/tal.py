from .base import RewardBase
from ..planner.purePursuit import PurePursuitPlanner

class TALReward(RewardBase):
    def __init__(self, map_manager, steer_range, speed_range, steer_w=0.4, speed_w=0.4, bias= 0.25, ratio=1.0, planner_cfg=None):
        super().__init__()
        self.map_manager = map_manager
        self.steer_range = steer_range
        self.speed_range = speed_range
        self.steer_w = steer_w
        self.speed_w = speed_w
        self.bias = bias
        self.ratio = ratio

        self.planner = PurePursuitPlanner(wheelbase=planner_cfg.wheelbase,
                                          map_manager=map_manager,
                                          lookahead=planner_cfg.lookahead, 
                                          gain=planner_cfg.gain,
                                          max_reacquire=planner_cfg.max_reacquire) 

    def get_reward(self, pre_obs, obs, action):
        base_reward = super().get_reward(obs, pre_obs)

        pp_action = self.planner.plan(pre_obs, id=0)
        
        
        steer_reward =  (abs(pp_action[0] - action[0]) / self.steer_range)  * self.steer_w
        throttle_reward =   (abs(pp_action[1] - action[1]) / self.speed_range) * self.speed_w

        reward = self.bias - steer_reward - throttle_reward
        reward = max(reward, 0) # limit at 0

        reward *= self.ratio
        return base_reward + reward