import os
import csv
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from f1tenth_gym.maps.map_manager import MapManager, MAP_DICT
from src.envs.envs import make_env
from src.planner.purePursuit import PurePursuitPlanner
from src.utils.helper import ScanBuffer, convert_scan, convert_action
from src.agents.agent import get_agent

@hydra.main(config_path="config", config_name="benchmark", version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    # --- 環境／プランナ／エージェント等の初期化 ---
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
    
    # --- モデルの読み込み ---
    agent = get_agent(agent_cfg=cfg.agent, device=device)
    model_path = os.path.join(cfg.ckpt)
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    
    scan_buffer = ScanBuffer(
        frame_size=cfg.envs.num_beams,
        num_scan=cfg.scan_n,
        target_size=cfg.downsample_beam
    )


    # --- ベンチマーク結果の保存ディレクトリ ---
    benchmark_dir = cfg.benchmark_dir
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)

    for ep in range(len(MAP_DICT)):
        map_name = MAP_DICT[ep]
        print(f"Evaluating on map: {map_name}")
        
        # マップごとのディレクトリとCSVファイルの準備
        map_dir = os.path.join(benchmark_dir, map_name)
        os.makedirs(map_dir, exist_ok=True)
        csv_file = os.path.join(map_dir, f"{map_name}_trajectory.csv")
        lap_file = os.path.join(map_dir, f"{map_name}_lap_times.csv")
        
        # CSVファイルの初期化
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y", "velocity"])

        # ラップタイムのCSV初期化
        if not os.path.exists(lap_file):
            with open(lap_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Lap Number", "Lap Time"])


        env.update_map(map_name, map_ext=cfg.envs.map.ext)
        scan_buffer.reset()
        # --- 評価ループ ---
        obs, info = env.reset()
        done = False

        # 初期スキャンの登録
        scan = convert_scan(obs['scans'][0], cfg.envs.max_beam_range)
        scan_buffer.add_scan(scan)
        total_reward = 0.0

        for step in range(cfg.num_steps):
            # 行動選択（エージェントとプランナ）
            actions = []
            for i in range(cfg.envs.num_agents):
                if i == 0:
                    state = scan_buffer.get_concatenated_tensor()
                    nn_action_dict = agent.select_action(state, evaluate=True)
                    nn_action = nn_action_dict['action']
                    action = convert_action(nn_action, steer_range=cfg.envs.steer_range, speed_range=cfg.envs.speed_range)
                else:
                    action = planner.plan(obs, id=i)
                actions.append(action)

            # 環境のステップ
            next_obs, reward, terminated, truncated, info = env.step(np.array(actions))
            terminated = obs['lap_counts'][0] == 1
            truncated = obs["collisions"][0]
            done = terminated or truncated

            # スキャンの更新
            next_scan = convert_scan(next_obs['scans'][0], cfg.envs.max_beam_range)
            scan_buffer.add_scan(next_scan)

            total_reward += reward

            # --- CSVへの書き込み ---
            current_pos = info['current_pos']   # [x, y]
            velocity = info['velocity']         # float
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_pos[0], current_pos[1], velocity])

            if done:
                # ラップタイム記録
                if truncated:
                    with open(lap_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([map_name, 0, 0.0])
                elif terminated:
                    lap_time = obs['lap_times']
                    with open(lap_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([map_name, 1, lap_time])

                break

            if cfg.render:
                env.render(cfg.render_mode)

            obs = next_obs

    print(f"Evaluation completed. Total Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()
