import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast,GradScaler
from .mlp_network import MlpNetwork
from .mlp_armpi_dataset import MlpArmpiDataset
from common.trainer import Trainer
from torch.utils.data import Dataset, DataLoader, random_split

class MlpTrain(Trainer):
    def __init__(self,args):
        super().__init__(args)

    def set_up(self,task_name):
        full_dataset = MlpArmpiDataset([task_name])
        self.model = MlpNetwork(state_input_dim=len(full_dataset.state_columns), action_output_dim=len(full_dataset.action_columns))
        self.scaler = GradScaler(enabled= (self.device.type == 'cuda'))
        return full_dataset 

    def _train(self):
        self.model.train()
        train_loss = 0.0
        batch_pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images,states, actions in batch_pbar:
            images = images.to(self.device)
            states = states.to(self.device)
            actions = actions.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type,enabled=(self.device.type == 'cuda')):
                outputs = self.model(states,images,None)
                loss = self.criterion(outputs, actions)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += loss.item() * images.size(0)

        return train_loss

    def _eval(self):
        self.model.eval()
        val_loss = 0.0
        batch_pbar = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, states, actions in batch_pbar:
                images = images.to(self.device)
                states = states.to(self.device)
                actions = actions.to(self.device)

                with autocast(device_type=self.device.type,enabled=(self.device.type == 'cuda')):
                    outputs = self.model(states,images,None)
                    loss = self.criterion(outputs, actions)

                val_loss += loss.item() * images.size(0)
        return val_loss
