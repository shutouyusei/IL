import torch
from train.train import Train
from networks.mlp_network import MlpNetwork
from data_edit.ArmpiDataset import ArmpiDataset
from torch.utils.data import Dataset, DataLoader, random_split

VAL_SPLIT = 0.2
BATCH_SIZE = 32

if __name__ == '__main__':
    read_data_list = ["bring_up_red"]
    full_dataset = ArmpiDataset(read_data_list)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MlpNetwork(state_input_dim=len(full_dataset.state_columns), action_output_dim=len(full_dataset.action_columns))

    trainer = Train(train_loader,val_loader,model,"model/result/model.pt",epochs=10)
    trainer.train()
