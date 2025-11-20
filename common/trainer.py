import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast,GradScaler
from pathlib import Path

class Trainer:
    def __init__(self,args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model_path = f"models/{args.file_name}.pt"

        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.patience = args.early_stop_patience
        self.criterion = nn.CrossEntropyLoss()
        full_dataset = self.set_up(args.task_name)

        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.to(self.device)

    def set_up(self,task_name):
        raise NotImplementedError

    def _train(self):
        raise NotImplementedError

    def _eval(self):
        raise NotImplementedError

    def train(self):
        best_val_loss = float('inf')
        early_stopping_counter = 0

        print("--- Training Start ---")
        epoch_pbar = tqdm(range(self.epochs), desc="Epochs")
        for epoch in epoch_pbar:
            train_loss = self._train()
            avg_train_loss = train_loss / len(self.train_loader.dataset)

            val_loss = self._eval()
            avg_val_loss = val_loss / len(self.val_loader.dataset)
            epoch_pbar.set_description(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                save_data = {
                    'model_state_dict': self.model.state_dict(),
                    'model_type': self.args.model,
                    'beset_val_loss': best_val_loss,
                    'epoch': epoch
                }
                torch.save(save_data, self.model_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter > self.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
                    break
        
        print("--- Training Finished ---")
        print(f"Best model saved to {self.model_path} (Val Loss: {best_val_loss:.6f})")


    
