import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from common.read_hdf import ReadHDF
from common.armpi_const import * # STATES_COLUMNS, ACTION_COLUMNS

class DiffusionPolicyDataset(Dataset):
    def __init__(self, data_directory_list, horizon=16, n_obs_steps=2):
        """
        Diffusion Policy用のデータセットクラス (ArmPi仕様)
        Args:
            data_directory_list (list): データディレクトリ名のリスト
            horizon (int): 予測する未来のステップ数 (入力履歴 + 出力未来)
            n_obs_steps (int): 観測として使用する過去のステップ数
        """
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.image_dataset_key = "images/data"
        
        self.action_columns = ACTION_COLUMNS
        self.state_columns = STATES_COLUMNS

        # 画像の前処理: Diffusion Policyは [-1, 1] の範囲を好むことが多い
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)), # Diffusion Policyのデフォルトサイズ例
            T.ToTensor(), # [0, 1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # [0, 1] -> [-1, 1]
        ])

        # データの読み込み
        self.episodes = self.__read_file(data_directory_list)
        
        # 統計量の計算 (正規化用)
        self.stats = self.compute_stats()
        
        # インデックスマップの作成
        self.index_map = [] 
        for ep_idx, df in enumerate(self.episodes):
            # エピソードの長さ
            episode_len = len(df)
            # 有効な開始インデックスの範囲
            # Diffusion Policyでは、現在のステップから horizon 分の未来が必要
            # また、n_obs_steps 分の過去も必要 (パディングで対応可能だが、ここでは単純化)
            for i in range(episode_len):
                # 単純に全ステップを学習データとする (終端処理は__getitem__で行う)
                self.index_map.append((ep_idx, i))

        print(f"Data read from {len(self.episodes)} episodes.")
        print(f"Total samples: {len(self.index_map)}")

    def __read_file(self, folder_name_list):
        read_hdf = ReadHDF()
        df_list = []
        for folder_name in folder_name_list:
            # 各エピソードのDataFrameリストを取得
            df_list.extend(read_hdf.read_hdf(folder_name))
        return df_list

    def compute_stats(self):
        """全データの統計量を計算する"""
        all_states = []
        all_actions = []
        for df in self.episodes:
            all_states.append(df[self.state_columns].values)
            all_actions.append(df[self.action_columns].values)
        
        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)

        stats = {
            "state_min": all_states.min(axis=0),
            "state_max": all_states.max(axis=0),
            "action_min": all_actions.min(axis=0),
            "action_max": all_actions.max(axis=0),
        }
        return stats

    def normalize_data(self, data, stats, key_prefix):
        """[-1, 1] に正規化する"""
        min_val = stats[f"{key_prefix}_min"]
        max_val = stats[f"{key_prefix}_max"]
        # ゼロ除算を防ぐ
        denom = max_val - min_val
        denom[denom < 1e-6] = 1.0
        
        return 2 * (data - min_val) / denom - 1

    def unnormalize_data(self, data, stats, key_prefix):
        """[-1, 1] から元のスケールに戻す"""
        min_val = stats[f"{key_prefix}_min"]
        max_val = stats[f"{key_prefix}_max"]
        return (data + 1) / 2 * (max_val - min_val) + min_val

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ep_idx, start_time_idx = self.index_map[idx]
        episode_df = self.episodes[ep_idx]
        episode_len = len(episode_df)

        time_idxes = np.arange(start_time_idx, start_time_idx + self.horizon)
        time_idxes = np.clip(time_idxes, 0, episode_len - 1)

        state_seq = episode_df.iloc[time_idxes][self.state_columns].values.astype(np.float32)
        action_seq = episode_df.iloc[time_idxes][self.action_columns].values.astype(np.float32)

        current_row = episode_df.iloc[start_time_idx]
        h5_file_path = current_row["file_path"]
        img_index = current_row["img_index"]
        
        try:
            with h5py.File(h5_file_path, "r") as hf:
                image_data = hf[self.image_dataset_key][img_index]
                image_tensor = self.transform(image_data) # [3, H, W]
                image_tensor = image_tensor.unsqueeze(0)# [1, 3, 224, 224]
                image_tensor = image_tensor.repeat(self.horizon, 1, 1, 1)
        except Exception as e:
            print(f"ERROR reading image: {e}")
            image_tensor = torch.zeros(self.horizon,3, 224, 224)

        
        data = {
            "obs": {
                "state": torch.tensor(state_seq, dtype=torch.float32), # [T, D_state]
                "primary": image_tensor, # [C, H, W]
            },
            "action": torch.tensor(action_seq, dtype=torch.float32) # [T, D_action]
        }

        return data
