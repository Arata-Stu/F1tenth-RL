import os
import math
import yaml
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from datetime import datetime


def visualize_speed_map(map_manager, figsize=(10,10), save_path=None):
    """YAML指定＋origin/resolution考慮でマップ上に速度可視化"""
    # YAML からマップ設定読み込み
    with open(map_manager.map_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    image_file = cfg['image']
    resolution = cfg['resolution']  # [m/pixel]
    origin_x, origin_y = cfg['origin'][0], cfg['origin'][1]

    # 画像読み込み
    img_path = os.path.join(map_manager.map_base_dir, image_file)
    img = plt.imread(img_path)
    height, width = img.shape[:2]

    # ワールド座標→ピクセル座標変換（Y軸反転補正）
    wpts = map_manager.waypoints[:, :2]
    px = (wpts[:, 0] - origin_x) / resolution
    py = height - (wpts[:, 1] - origin_y) / resolution

    # 描画
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin='upper')
    sc = ax.scatter(px, py, c=map_manager.waypoints[:, 2], cmap='bwr', s=10)

    # カラーバー
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Speed [m/s]')
    ax.set_title(f"Speed Visualization: {map_manager.map_name}")
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def visualize_curve_class_map(map_manager, figsize=(10,10), save_path=None):
    """マップ上にカーブのクラス分類結果を可視化（速度ではなくクラスで色分け）"""
    # YAML からマップ設定読み込み
    with open(map_manager.map_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    image_file = cfg['image']
    resolution = cfg['resolution']  # [m/pixel]
    origin_x, origin_y = cfg['origin'][0], cfg['origin'][1]

    # 画像読み込み
    img_path = os.path.join(map_manager.map_base_dir, image_file)
    img = plt.imread(img_path)
    height, width = img.shape[:2]

    # ワールド座標→ピクセル座標変換（Y軸反転）
    wpts = map_manager.waypoints[:, :2]
    px = (wpts[:, 0] - origin_x) / resolution
    py = height - (wpts[:, 1] - origin_y) / resolution

    # クラス情報
    if not hasattr(map_manager, 'curve_classes'):
        raise AttributeError("MapManagerに 'curve_classes' が存在しません。_compute_speeds で分類が行われているか確認してください。")
    classes = map_manager.curve_classes

    # 描画
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin='upper')
    sc = ax.scatter(px, py, c=classes, cmap='RdYlBu', s=10)

    # カラーバー
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(classes.min(), classes.max()+1))
    cbar.set_label('Curve Class (0=Straight, ↑Tighter)')
    ax.set_title(f"Curve Class Visualization: {map_manager.map_name}")
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


def visualize_trajectory(MAP_DICT, base_dir='../benchmarks/2025-05-10/18-54-23/', yaml_dir='./f1tenth_gym/maps'):
    '''
    全てのトラジェクトリを個別に描画し、最終的に1枚の画像に結合して表示
    
    Args:
        MAP_DICT (dict): マップIDと名前の辞書
        base_dir (str): CSVファイルが存在するベースディレクトリのパス
        yaml_dir (str): YAMLファイルが存在するベースディレクトリのパス
    '''
    print(f"MAP_DICT contains {len(MAP_DICT)} maps")

    # --- 現在の日付と時刻を取得してパスを作成 ---
    timestamp = datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
    output_dir = os.path.join('outputs', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    image_list = []
    cols = 4  # グリッドの列数
    rows = math.ceil(len(MAP_DICT) / cols)  # 必要な行数を計算

    # --- 各トラックを描画 ---
    for key, map_name in MAP_DICT.items():
        print(f'Processing: {map_name}')

        # パスの構築
        yaml_path = os.path.join(yaml_dir, map_name, f'{map_name}_map.yaml')
        csv_path = os.path.join(base_dir, map_name, f'{map_name}_trajectory.csv')

        if not os.path.exists(yaml_path) or not os.path.exists(csv_path):
            print(f'Warning: Skipping {map_name} due to missing files.')
            continue

        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        image_file = cfg['image']
        resolution = cfg['resolution']
        origin_x, origin_y = cfg['origin'][0], cfg['origin'][1]

        img_path = os.path.join(yaml_dir, map_name, image_file)
        if not os.path.exists(img_path):
            print(f'Warning: Image not found for {map_name}. Skipping visualization.')
            continue

        img = plt.imread(img_path)
        height, width = img.shape[:2]

        x_coords, y_coords = [], []

        with open(csv_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            reader.fieldnames = [header.strip() for header in reader.fieldnames]
            for row in reader:
                x_coords.append(float(row['x']))
                y_coords.append(float(row['y']))

        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)

        px = (x_coords - origin_x) / resolution
        py = height - (y_coords - origin_y) / resolution

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, origin='upper')
        ax.plot(px, py, color='red', linewidth=2)
        ax.set_title(map_name, fontsize=12)
        ax.axis('off')

        individual_save_path = os.path.join(output_dir, f'{map_name}.png')
        plt.savefig(individual_save_path, format='png', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

        image_list.append(Image.open(individual_save_path))

    thumbnail_size = (400, 400)
    for img in image_list:
        img.thumbnail(thumbnail_size)

    grid_width = cols * 400
    grid_height = rows * 400
    merged_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

    for index, image in enumerate(image_list):
        x = (index % cols) * 400
        y = (index // cols) * 400
        merged_image.paste(image, (x, y))

    merged_save_path = os.path.join(output_dir, 'merged_trajectory.png')
    merged_image.save(merged_save_path)

    print(f'Images saved to {output_dir}')