import torch
import torch.nn as nn
import torch.optim as optim
from common.trainer import Trainer
from .act import build_ACT 
from .act_armpi_dataset import ActArmpiDataset
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast,GradScaler
from tqdm import tqdm

class ActTrainer(Trainer):
    def __init__(self,args):
        super().__init__(args)

    def set_up(self,task_name):
        full_dataset = ActArmpiDataset([task_name])

        self.model = build_ACT(state_dim=len(full_dataset.state_columns),action_dim=len(full_dataset.action_columns))
        return full_dataset

    def _train(self):
        self.model.train()
        train_loss = 0.0
        batch_pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images,states, actions, is_pad in batch_pbar:
            images = images.to(self.device)
            states = states.to(self.device)
            actions = actions.to(self.device)
            is_pad =  is_pad.to(self.device)
            if images.dim() == 4 and images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
                        
            if images.dim() == 4:
                images = images.unsqueeze(1)
            self.optimizer.zero_grad()
            a_hat,is_pad_hat, (mu,logvar) = self.model(states,images,None,actions,is_pad)
            l1_loss = self.criterion(a_hat, actions)
                
            total_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss_kld = total_kld / images.shape[0] # バッチサイズで割る
            
            kl_weight = getattr(self, 'kl_weight', 10.0) 
            loss = l1_loss + (kl_weight * loss_kld)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            train_loss += loss.item() * images.size(0)

        return train_loss

    def _eval(self):
        self.model.eval()
        val_loss = 0.0
        batch_pbar = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, states, actions, is_pad in batch_pbar:
                images = images.to(self.device)
                states = states.to(self.device)
                actions = actions.to(self.device)
                is_pad =  is_pad.to(self.device)

                if images.dim() == 4 and images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                            
                if images.dim() == 4:
                    images = images.unsqueeze(1)
                a_hat,is_pad_hat, (mu,logvar) = self.model(states,images,None,actions,is_pad)
                l1_loss = self.criterion(a_hat, actions)
                    
                total_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss_kld = total_kld / images.shape[0] # バッチサイズで割る
                    
                kl_weight = getattr(self, 'kl_weight', 10.0) 
                loss = l1_loss + (kl_weight * loss_kld)

                val_loss += loss.item() * images.size(0)
        return val_loss
