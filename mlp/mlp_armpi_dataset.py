import h5py
import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from common.read_hdf import ReadHDF


class MlpArmpiDataset(Dataset):
    def __init__(self, data_directory_list):
        all_sync_data = self.__read_file(data_directory_list)
        print(f"{len(all_sync_data)} files are read")


        self.image_dataset_key = "images/data"
        self.action_columns = [
            "chassis_move_forward",
            "chassis_move_right",
            "angular_right",
            "arm_x",
            "arm_y",
            "arm_z",
            "arm_alpha",
            "rotation",
            "gripper_close",
        ]
        self.state_columns = [
            "joint1_pos",
            "joint2_pos",
            "joint3_pos",
            "joint4_pos",
            "joint5_pos",
            "r_joint_pos",
        ]
        # TODO: 画像の処理方法見直す
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.master_index = pd.concat(all_sync_data, ignore_index=True)
        original_count = len(self.master_index)
        action_sum = self.master_index[self.action_columns].abs().sum(axis=1)

        self.master_index = self.master_index[action_sum > 0]
        print(f"{original_count - len(self.master_index)} data are removed")

    def __read_file(self, folder_name_list):
        read_hdf = ReadHDF()
        df_synced_list = []
        for folder_name in folder_name_list:
            df_synced_list += read_hdf.read_hdf(folder_name)
        return df_synced_list

    def __len__(self):
        return len(self.master_index)

    def __getitem__(self, idx):
        metadata_row = self.master_index.iloc[idx]
        
        # action data
        action_values = metadata_row[self.action_columns].values.astype(np.int64) 
        action_labels = action_values + 1 
        action_tensor = torch.tensor(action_labels, dtype=torch.long)

        # state data
        state_values = metadata_row[self.state_columns].values.astype(np.float32)
        state_tensor = torch.tensor(state_values, dtype=torch.float32)

        # image data
        h5_file_path = metadata_row["file_path"]
        img_index = metadata_row["img_index"]
        try:
            with h5py.File(h5_file_path, "r") as hf:
                image_data = hf[self.image_dataset_key][img_index]
        except Exception as e:
            print(f"ERROR: {e}")
            return (
                torch.randn(3, 224, 224),
                torch.zeros(len(self.state_columns)),
                torch.zeros(len(self.action_columns)),
            )

        # preprocess image
        image_tensor = self.transform(image_data)

        return image_tensor, state_tensor, action_tensor


if __name__ == "__main__":
    print("TEST")
    armpi_dataset = ArmpiDataset( ["bring_up_red"])
    for i in range(len(armpi_dataset)):
        image_tensor, state_tensor, action_tensor = armpi_dataset[i]
