import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from common.read_hdf import ReadHDF

class ActArmpiDataset(Dataset):
    def __init__(self, data_directory_list, num_queries=100):
        """
        ACTモデル用のデータセットクラス
        Args:
            data_directory_list (list): データディレクトリ名のリスト
            num_queries (int): 予測するアクションのチャンクサイズ (例: 100)
        """
        self.num_queries = num_queries
        self.image_dataset_key = "images/data"
        
        self.action_columns = [
            "chassis_move_forward", "chassis_move_right", "angular_right",
            "arm_x", "arm_y", "arm_z", "arm_alpha", "rotation", "gripper_close"
        ]
        self.state_columns = [
            "joint1_pos", "joint2_pos", "joint3_pos", "joint4_pos",
            "joint5_pos", "r_joint_pos"
        ]

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.episodes = self.__read_file(data_directory_list)
        self.index_map = [] # (episode_idx, row_idx) のリスト

        total_original_rows = 0
        total_kept_rows = 0

        # 各エピソード(ファイル)ごとに処理
        for ep_idx, df in enumerate(self.episodes):
            total_original_rows += len(df)
            
            action_sum = df[self.action_columns].abs().sum(axis=1)
            valid_indices = np.where(action_sum > 0)[0].tolist()
            
            # 有効な行のみを学習データとして登録
            for row_idx in valid_indices:
                self.index_map.append((ep_idx, row_idx))
            
            total_kept_rows += len(valid_indices)

        print(f"Data Loaded: {len(self.episodes)} episodes.")
        print(f"Total frames: {total_kept_rows} (Removed {total_original_rows - total_kept_rows} idle frames)")

    def __read_file(self, folder_name_list):
        # ReadHDFを使って読み込むが、pd.concatせずにリストのまま保持する
        read_hdf = ReadHDF()
        df_list = []
        for folder_name in folder_name_list:
            # read_hdf.read_hdf はリスト[df, df, ...] を返すと想定
            dfs = read_hdf.read_hdf(folder_name)
            df_list.extend(dfs)
        return df_list

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 1. 今回取得するエピソードIDと、その中の行番号を特定
        ep_idx, start_row_idx = self.index_map[idx]
        episode_df = self.episodes[ep_idx]
        
        # 2. メタデータの取得 (現在のステップ)
        current_row = episode_df.iloc[start_row_idx]
        
        # --- State Data (qpos) ---
        state_values = current_row[self.state_columns].values.astype(np.float32)
        qpos_tensor = torch.tensor(state_values, dtype=torch.float32)

        # --- Image Data ---
        h5_file_path = current_row["file_path"]
        img_index = current_row["img_index"]
        
        try:
            with h5py.File(h5_file_path, "r") as hf:
                image_data = hf[self.image_dataset_key][img_index]
                image_tensor = self.transform(image_data) # [3, 224, 224]
        except Exception as e:
            print(f"ERROR reading image: {e}")
            # エラー時はダミーデータ
            image_tensor = torch.zeros(3, 224, 224)

        # --- Action Sequence (Chunking) ---
        # 現在の行から num_queries 分だけ未来のアクションを取得する
        # エピソードの範囲を超えないようにスライスする
        max_len = len(episode_df)
        end_row_idx = min(start_row_idx + self.num_queries, max_len)
        
        # スライスで取得
        action_chunk_df = episode_df.iloc[start_row_idx : end_row_idx][self.action_columns]
        action_values = action_chunk_df.values.astype(np.float32)
        
        # パディング処理
        # 取得できた長さ
        current_seq_len = len(action_values)
        # 足りない長さ
        pad_len = self.num_queries - current_seq_len
        
        if pad_len > 0:
            # 最後の値を繰り返してパディング (エッジパディング)
            last_action = action_values[-1]
            padding = np.tile(last_action, (pad_len, 1))
            action_values = np.concatenate([action_values, padding], axis=0)
            
            # パディングマスク (0: データあり, 1: パディング)
            is_pad = np.concatenate([
                np.zeros(current_seq_len, dtype=bool),
                np.ones(pad_len, dtype=bool)
            ])
        else:
            is_pad = np.zeros(self.num_queries, dtype=bool)

        actions_tensor = torch.tensor(action_values, dtype=torch.float32)
        is_pad_tensor = torch.tensor(is_pad, dtype=torch.bool)

        return image_tensor, qpos_tensor, actions_tensor, is_pad_tensor

if __name__ == "__main__":
    print("TEST ActArmpiDataset")
    # ディレクトリ名はご自身の環境に合わせて変更してください
    dataset = ActArmpiDataset(["bring_up_red"], num_queries=100)
    
    if len(dataset) > 0:
        img, qpos, actions, is_pad = dataset[0]
        print(f"Image shape: {img.shape}")       # [3, 224, 224]
        print(f"Qpos shape: {qpos.shape}")       # [6]
        print(f"Actions shape: {actions.shape}") # [100, 9]
        print(f"Is_pad shape: {is_pad.shape}")   # [100]
        print("Test Passed!")
    else:
        print("No data found.")
