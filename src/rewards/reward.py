from omegaconf import DictConfig

from .progress import ProgressReward
from .tal import TALReward
from f1tenth_gym.maps.map_manager import MapManager

def make_raward(reward_cfg: DictConfig, map_manager: MapManager=None):
    reward_name = reward_cfg.name
    if reward_name == "progress":
        return ProgressReward(ratio=reward_cfg.ratio, map_manager=map_manager)
    elif reward_name == "TAL":
        return TALReward(
            map_manager=map_manager,
            steer_range=reward_cfg.steer_range,
            speed_range=reward_cfg.speed_range,
            steer_w=reward_cfg.steer_w,
            speed_w=reward_cfg.speed_w,
            bias=reward_cfg.bias,
            ratio=reward_cfg.ratio,
            planner_cfg=reward_cfg.planner,

        )
    else:
        raise ValueError(f"Invalid reward type: {reward_name}")