import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from common.trainer import Trainer
from common.armpi_const import *

from diffusion.diffusion_network import build_diffusion_policy
from diffusion.diffusion_armpi_dataset import DiffusionPolicyDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusers.optimization import get_scheduler

class DiffusionTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=1e-6
        )

        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500, # ウォームアップステップ
            num_training_steps=len(self.train_loader) * self.epochs
        )

    def set_up(self, task_name):
        full_dataset = DiffusionPolicyDataset([task_name], horizon=16)

        self.model = build_diffusion_policy(
            state_dim=len(STATES_COLUMNS),
            action_dim=len(ACTION_COLUMNS) 
        )
        all_states = []
        all_actions = []

        
        for df in full_dataset.episodes:
            all_states.append(df[full_dataset.state_columns].values)
            all_actions.append(df[full_dataset.action_columns].values)
            
        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        return full_dataset

    def _train(self):
        self.model.train()
        train_loss = 0.0
        
        batch_pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in batch_pbar:
            nbatch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

            self.optimizer.zero_grad()
            loss = self.model.compute_loss(nbatch)


            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            loss_val = loss.item()
            train_loss += loss_val
            
            batch_pbar.set_postfix(loss=loss_val)

        return train_loss

    def _eval(self):
        self.model.eval()
        val_loss = 0.0
        
        batch_pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch in batch_pbar:
                nbatch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                loss = self.model.compute_loss(nbatch)
                val_loss += loss.item()

        return val_loss
