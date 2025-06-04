import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.actor import get_actor
from src.data.dataset.lidar_dataset import MultiBagE2EDataset
from src.data.dataset.transform import E2ETransform
from src.utils.helper import convert_action

@hydra.main(config_path="config", config_name="train_bc", version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    # --- 設定 ---
    input_length = cfg.model.input_length
    range_max = cfg.data.range_max
    batch_size = cfg.train.batch_size
    num_epochs = cfg.train.num_epochs
    lr = cfg.train.lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 保存先作成 ---
    save_path = cfg.train.save_dir
    os.makedirs(save_path, exist_ok=True)

    # --- データセットとローダー ---
    transform = E2ETransform(range_max=range_max, base_num=1081, downsample_num=input_length)
    dataset = MultiBagE2EDataset(root_dir=cfg.data.root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Actorモデルの初期化とロード ---
    actor = get_actor(state_dim=input_length,
                      action_dim=2,
                      hidden_dim=cfg.model.hidden_dim,
                      policy_type=cfg.model.policy_type).to(device)
    
    if cfg.model.pretrained_path:
        checkpoint = torch.load(cfg.model.pretrained_path, map_location=device)
                
                # 'actor' というキーで保存されているアクターの state_dict を読み込む
        if 'actor' in checkpoint:
            actor.load_state_dict(checkpoint['actor'])
            print(f"[✔] Pretrained actor weights successfully loaded from {cfg.model.pretrained_path}")

            for param in actor.lidar_backbone.parameters():
                param.requires_grad = False
        else:
            
            print(f"[!] Warning: 'actor' key not found in the checkpoint. Attempting to load the entire file as state_dict...")
            ## error
            NotImplementedError("The checkpoint does not contain the expected 'actor' key. Please check the file format.")

    # --- 最適化設定 ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(actor.parameters(), lr=lr)

    # --- 教師あり再学習ループ ---
    for epoch in range(num_epochs):
        actor.train()
        running_loss = 0.0
        for batch in loader:
            scan = batch['scan'].to(device)
            target = torch.stack([batch['steer'], batch['speed']], dim=1).to(device)

            mean, _ = actor(scan)
            action = torch.tanh(mean)
            action = convert_action(action, steer_range=1.0, speed_range=1.0)

            loss = criterion(action, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # チェックポイント保存
        torch.save(actor.state_dict(), os.path.join(save_path, f'finetuned_actor_epoch{epoch+1}.pth'))

if __name__ == "__main__":
    main()
