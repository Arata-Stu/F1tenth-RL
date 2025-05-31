import os
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from f1tenth_gym.maps.map_manager import MapManager, MAP_DICT
from src.envs.envs import make_env
from src.rewards.reward import make_raward
from src.buffers.buffer import get_buffer
from src.agents.agent import get_agent
from src.planner.purePursuit import PurePursuitPlanner
from src.utils.helper import ScanBuffer, convert_scan, convert_action

@hydra.main(config_path="config", config_name="train_rl", version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    # --- 環境／プランナ／エージェント等の初期化（省略） ---
    map_manager = MapManager(
        map_name=cfg.envs.map.name,
        map_ext=cfg.envs.map.ext,
        speed=cfg.envs.map.speed,
        downsample=cfg.envs.map.downsample,
        use_dynamic_speed=cfg.envs.map.use_dynamic_speed,
        a_lat_max=cfg.envs.map.a_lat_max,
        smooth_sigma=cfg.envs.map.smooth_sigma
    )
    env = make_env(cfg.envs, map_manager, cfg.vehicle)
    planner = PurePursuitPlanner(
        wheelbase=cfg.planner.wheelbase,
        map_manager=map_manager,
        lookahead=cfg.planner.lookahead,
        gain=cfg.planner.gain,
        max_reacquire=cfg.planner.max_reacquire,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    agent = get_agent(agent_cfg=cfg.agent, device=device)
    buffer = get_buffer(buffer_cfg=cfg.buffer, device=device)
    reward_manager = make_raward(reward_cfg=cfg.reward, map_manager=map_manager)
    scan_buffer = ScanBuffer(
        frame_size=cfg.envs.num_beams,
        num_scan=cfg.scan_n,
        target_size=cfg.downsample_beam
    )

    log_dir = os.path.join(cfg.log_dir, cfg.agent.name, cfg.reward.name, cfg.run_id)
    model_dir = os.path.join(cfg.ckpt_dir, cfg.agent.name, cfg.reward.name, cfg.run_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    best_reward = -float("inf")  # 最高報酬の初期値
    warmup_steps = cfg.warmup_steps
    global_step = 0  # 全エピソードを通じたステップ数カウンタ

    # --- 学習ループ ---
    for episode in range(cfg.num_episodes):
        if cfg.random_map:
            map_name = random.choice(list(MAP_DICT.values()))
            env.update_map(map_name, map_ext=cfg.envs.map.ext)
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        # 初期スキャン登録
        scan = convert_scan(obs['scans'][0], cfg.envs.max_beam_range)
        scan_buffer.add_scan(scan)

        for step in range(cfg.num_steps):
            global_step += 1
            # 行動選択（エージェントとプランナ）
            actions = []
            for i in range(cfg.envs.num_agents):
                if i == 0:
                    state = scan_buffer.get_concatenated_tensor()
                    nn_action_dict = agent.select_action(state, evaluate=False)
                    nn_action = nn_action_dict["action"]
                    log_prob = nn_action_dict["log_prob"]
                    action = convert_action(nn_action, steer_range=cfg.envs.steer_range, speed_range=cfg.envs.speed_range)
                else:
                    action = planner.plan(obs, id=i)
                actions.append(action)

            next_obs, reward, terminated, truncated, info = env.step(np.array(actions))
            done = terminated or truncated

            # スキャン更新
            next_scan = convert_scan(next_obs['scans'][0], cfg.envs.max_beam_range)
            scan_buffer.add_scan(next_scan)

            # 報酬計算・バッファ追加
            r = reward_manager.get_reward(obs=next_obs, pre_obs=obs, action=actions[0])
            total_reward += r
            next_state = scan_buffer.get_concatenated_numpy()
    
            buffer.add(state, action, r, next_state, done)

            # 学習ステップ
            ## バッファのサイズが十分であれば学習を行う
            if global_step >= warmup_steps and len(buffer) > cfg.batch_size:
                loss_dict = agent.update(buffer, cfg.batch_size)
                for key, value in loss_dict.items():
                    writer.add_scalar(f"loss/{key}", value, global_step=episode)

            obs = next_obs

            if cfg.render:
                env.render(cfg.render_mode)
                
            if done:
                break

        # エピソード終了後のログ
        writer.add_scalar("reward/total_reward", total_reward, global_step=episode)
        print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

        # --- 最高報酬更新＆モデル保存 ---
        if total_reward > best_reward:
            best_reward = total_reward
            save_path = os.path.join(model_dir, f"best_model_reward_{best_reward:.2f}.pth")
            agent.save(save_path)
            print(f"  ▶ New best model saved (reward: {best_reward:.2f}) → {save_path}")

    writer.close()
    env.close()
    print("Training completed.")

if __name__ == "__main__":
    main()