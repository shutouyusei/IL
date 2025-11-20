import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from common.read_hdf import ReadHDF
from common.armpi_const import * # STATES_COLUMNS, ACTION_COLUMNS を使用

class ActArmpiDataset(Dataset):
    def __init__(self, data_directory_list, num_queries=100): # num_queries を引数に追加
        self.num_queries = num_queries
        
        # NOTE: MLPのデータセットとは異なり、エピソード境界をまたがないように処理が必要です
        # read_hdf が返すのは「エピソードのリスト」であると仮定し、pd.concat はしません。
        self.episodes = self.__read_file(data_directory_list)
        
        print(f"Data read from {len(self.episodes)} files/episodes.")

        self.image_dataset_key = "images/data"
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- 全エピソードをフラットにして、インデックスマップを作成 ---
        self.index_map = [] # (episode_idx, row_idx) のリスト
        total_original_rows = 0
        total_kept_rows = 0

        for ep_idx, df in enumerate(self.episodes):
            total_original_rows += len(df)
            
            # --- 静止データの除去 ---
            action_sum = df[ACTION_COLUMNS].abs().sum(axis=1)
            # np.where を使って物理的な行番号を取得 (以前の Indexing エラー対策)
            valid_rows = np.where(action_sum > 0)[0].tolist() 
            
            
            for row_idx in valid_rows:
                self.index_map.append((ep_idx, row_idx))
            
            total_kept_rows += len(valid_rows)

        print(f"Total frames kept: {total_kept_rows} (Removed {total_original_rows - total_kept_rows} idle frames)")


    def __read_file(self, folder_name_list):
        read_hdf = ReadHDF()
        df_list = []
        for folder_name in folder_name_list:
            # ReadHDFが各ファイル/タスクからDFのリストを返すことを想定
            df_list.extend(read_hdf.read_hdf(folder_name))
        return df_list

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ep_idx, start_row_idx = self.index_map[idx]
        episode_df = self.episodes[ep_idx]
        
        # 1. 現在のメタデータ (画像/状態)
        current_row = episode_df.iloc[start_row_idx]
        
        # --- Image Data (処理はMLPと同じ) ---
        h5_file_path = current_row["file_path"]
        img_index = current_row["img_index"]
        try:
            with h5py.File(h5_file_path, "r") as hf:
                image_data = hf[self.image_dataset_key][img_index]
                image_tensor = self.transform(image_data)
        except Exception as e:
            print(f"ERROR reading image: {e}")
            image_tensor = torch.zeros(3, 224, 224)


        # --- State Data (qpos) ---
        state_values = current_row[STATES_COLUMNS].values.astype(np.float32)
        qpos_tensor = torch.tensor(state_values, dtype=torch.float32)

        # ----------------------------------------------------
        # 2. アクションシーケンスの取得とパディング (ACTのコア部分)
        # ----------------------------------------------------
        max_len = len(episode_df)
        end_row_idx = min(start_row_idx + self.num_queries, max_len)
        
        # ilocによるスライスでアクションのチャンクを取得
        action_chunk_df = episode_df.iloc[start_row_idx : end_row_idx][ACTION_COLUMNS]
        
        action_values = action_chunk_df.values.astype(np.int64) 
        action_labels = action_values + 1
        
        # Padding 処理
        current_seq_len = len(action_labels)
        pad_len = self.num_queries - current_seq_len
        
        if pad_len > 0:
            # 最後の値を繰り返してパディング (エッジパディング)
            last_action = action_labels[-1]
            padding = np.tile(last_action, (pad_len, 1))
            action_labels = np.concatenate([action_labels, padding], axis=0)
            
            # パディングマスク (0: データあり, 1: パディング)
            is_pad = np.concatenate([
                np.zeros(current_seq_len, dtype=bool),
                np.ones(pad_len, dtype=bool)
            ])
        else:
            is_pad = np.zeros(self.num_queries, dtype=bool)
        is_pad_tensor = torch.tensor(is_pad, dtype=torch.bool)
        actions_tensor = torch.tensor(action_labels, dtype=torch.long)
        
        # 戻り値: image, qpos, actions (chunk), is_pad
        return image_tensor, qpos_tensor, actions_tensor, is_pad_tensor


if __name__ == "__main__":
    print("TEST ActArmpiDataset")
    # 例として num_queries=100 でテスト
    dataset = ActArmpiDataset(["bring_up_red"], num_queries=100)
    
    if len(dataset) > 0:
        img, qpos, actions, is_pad = dataset[0]
        print(f"Image shape: {img.shape}")       # [3, 224, 224]
        print(f"Qpos shape: {qpos.shape}")       # [6]
        print(f"Actions shape: {actions.shape}") # [100, 9] (Seq_Len, Action_Dim)
        print(f"Is_pad shape: {is_pad.shape}")   # [100]
        
        # 連続値 (float32) になっているか確認
        if actions.dtype == torch.float32:
            print("✅ Actions dtype is FLOAT32 (Correct for Regression)")
        else:
            print(f"❌ Actions dtype is {actions.dtype} (Expected float32)")
            
        print("Test Passed!")
    else:
        print("No data found.")
