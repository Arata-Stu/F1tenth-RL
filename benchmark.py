import os
import csv
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from f1tenth_gym.maps.map_manager import MapManager, MAP_DICT
# src以下のモジュールはユーザー環境に存在することを前提とします
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
    # cfg.ckpt がモデルファイルへのフルパスであることを想定
    model_path = cfg.ckpt
    if not os.path.isabs(model_path) and hydra.utils.get_original_cwd() != os.getcwd():
        # Hydraの作業ディレクトリ変更に対応するため、元のCWDからの相対パスとして解決
        model_path = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt)
        
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

    # --- 全マップのラップタイムを集約して保存するCSVファイルの準備 ---
    all_laps_summary_file = os.path.join(benchmark_dir, "all_maps_lap_summary.csv")
    with open(all_laps_summary_file, mode='w', newline='') as file:
        summary_writer = csv.writer(file)
        summary_writer.writerow(["Map Name", "Completion Status", "Lap Time (s)"])

    total_reward_all_maps = 0.0 # 全マップ、全ステップの累積報酬用

    for ep_idx, map_name_key in enumerate(MAP_DICT): # MAP_DICTのキーをイテレート
        map_name = MAP_DICT[map_name_key] # マップ名を取得
        print(f"Evaluating on map: {map_name} ({ep_idx+1}/{len(MAP_DICT)})")

        # マップごとのディレクトリとCSVファイルの準備
        map_dir = os.path.join(benchmark_dir, map_name)
        os.makedirs(map_dir, exist_ok=True)
        trajectory_csv_file = os.path.join(map_dir, f"{map_name}_trajectory.csv")
        map_lap_time_file = os.path.join(map_dir, f"{map_name}_lap_times.csv")

        # 軌跡用CSVファイルの初期化
        with open(trajectory_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y", "velocity"])

        # マップごとのラップタイムCSV初期化 (ファイルが存在しない場合のみヘッダー書き込み)
        if not os.path.exists(map_lap_time_file):
            with open(map_lap_time_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                # 元のコードのヘッダーと書き込み内容の不一致を維持
                writer.writerow(["Lap Number", "Lap Time"])


        env.update_map(map_name, map_ext=cfg.envs.map.ext)
        scan_buffer.reset()
        
        obs, info = env.reset()
        done = False

        scan = convert_scan(obs['scans'][0], cfg.envs.max_beam_range)
        scan_buffer.add_scan(scan)
        
        current_map_reward_sum = 0.0

        for step in range(cfg.num_steps):
            actions = []
            for i in range(cfg.envs.num_agents):
                if i == 0: # エージェント0 (通常はEgo Agent)
                    state = scan_buffer.get_concatenated_tensor()
                    nn_action_dict = agent.select_action(state, evaluate=True)
                    nn_action = nn_action_dict['action']
                    action = convert_action(nn_action, steer_range=cfg.envs.steer_range, speed_range=cfg.envs.speed_range)
                else: # 他のエージェント (プランナー制御)
                    action = planner.plan(obs, id=i) # 元のコード通り obs を渡す
                actions.append(action)

            next_obs, reward, env_terminated, env_truncated, info = env.step(np.array(actions))
            
            # エージェント0 (Ego Agent) の状態に基づいて終了判定
            agent0_completed_lap = next_obs['lap_counts'][0] >= 1
            agent0_collided = next_obs['collisions'][0]

            # ユーザー定義の終了条件
            # 1周完走 (terminated) または衝突 (truncated) でエピソード終了
            sim_terminated_by_completion = agent0_completed_lap
            sim_truncated_by_collision = agent0_collided
            
            done = sim_terminated_by_completion or sim_truncated_by_collision

            next_scan = convert_scan(next_obs['scans'][0], cfg.envs.max_beam_range)
            scan_buffer.add_scan(next_scan)

            # 報酬の加算 (エージェント0の報酬を想定、スカラーまたは配列の最初の要素)
            current_step_reward = reward[0] if isinstance(reward, (np.ndarray, list)) and len(reward) > 0 else reward
            current_map_reward_sum += current_step_reward
            total_reward_all_maps += current_step_reward


            # --- 軌跡CSVへの書き込み ---
            # infoからではなく、next_obsから位置と速度を取得するのがより正確
            current_pos_x = next_obs['poses_x'][0]
            current_pos_y = next_obs['poses_y'][0]
            velocity = next_obs['linear_vels_x'][0] 
            with open(trajectory_csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_pos_x, current_pos_y, velocity])

            if done:
                lap_time_value = 0.0
                completion_status_code = 0 # 0: DNF (e.g., collision), 1: Finished

                if sim_truncated_by_collision:
                    print(f"Map {map_name}: Agent 0 collided after {step + 1} steps.")
                    completion_status_code = 0
                    lap_time_value = 0.0 # 衝突時はラップタイム0
                    
                    # マップごとのラップタイムファイルへ記録 (元の形式を踏襲)
                    with open(map_lap_time_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([map_name, 0, 0.0])
                    
                    # 集約CSVファイルへ記録
                    with open(all_laps_summary_file, mode='a', newline='') as file:
                        summary_writer = csv.writer(file)
                        summary_writer.writerow([map_name, completion_status_code, lap_time_value])

                elif sim_terminated_by_completion:
                    # next_obs['lap_times'] は完了したラップのタイムのリスト。最初のものを取得。
                    if next_obs['lap_times'] and len(next_obs['lap_times']) > 0:
                        lap_time_value = next_obs['lap_times'][0]
                    else:
                        # 通常発生しないはずだが、フォールバック
                        lap_time_value = 0.0 
                    completion_status_code = 1
                    print(f"Map {map_name}: Agent 0 completed lap. Time: {lap_time_value:.3f}s")

                    # マップごとのラップタイムファイルへ記録 (元の形式を踏襲)
                    with open(map_lap_time_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([map_name, 1, lap_time_value])
                        
                    # 集約CSVファイルへ記録
                    with open(all_laps_summary_file, mode='a', newline='') as file:
                        summary_writer = csv.writer(file)
                        summary_writer.writerow([map_name, completion_status_code, lap_time_value])
                
                break # 現在のマップの評価ループを終了

            if cfg.render:
                env.render(cfg.render_mode)

            obs = next_obs
        
        print(f"Finished evaluation for map: {map_name}. Sum of rewards for this map: {current_map_reward_sum:.2f}")

    print(f"Evaluation completed for all maps. Total accumulated reward: {total_reward_all_maps:.2f}")
    print(f"All map lap times summary saved to: {all_laps_summary_file}")
    env.close()

if __name__ == "__main__":
    main()