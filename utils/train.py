import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast,GradScaler

class Train:
    def __init__(self,train_loader,val_loader,model,model_path,learning_rate = 0.001,epochs = 10,early_stop_patience = 10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(self.device)
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = early_stop_patience

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.scaler = GradScaler(enabled= (self.device.type == 'cuda'))

    def train(self):
        best_val_loss = float('inf')
        early_stopping_counter = 0

        print("--- Training Start ---")
        epoch_pbar = tqdm(range(self.epochs), desc="Epochs")
        for epoch in epoch_pbar:
            train_loss = self.__train()
            avg_train_loss = train_loss / len(self.train_loader.dataset)

            val_loss = self.__eval()
            avg_val_loss = val_loss / len(self.val_loader.dataset)
            epoch_pbar.set_description(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                torch.save(self.model, self.model_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter > self.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
                    break
        
        print("--- Training Finished ---")
        print(f"Best model saved to {self.model_path} (Val Loss: {best_val_loss:.6f})")


    
    def __train(self):
        self.model.train()
        train_loss = 0.0
        batch_pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images,states, actions in batch_pbar:
            images = images.to(self.device)
            states = states.to(self.device)
            actions = actions.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type,enabled=(self.device.type == 'cuda')):
                outputs = self.model(images,states)
                loss = self.criterion(outputs, actions)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += loss.item() * images.size(0)

        return train_loss

    def __eval(self):
        self.model.eval()
        val_loss = 0.0
        batch_pbar = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, states, actions in batch_pbar:
                images = images.to(self.device)
                states = states.to(self.device)
                actions = actions.to(self.device)

                with autocast(device_type=self.device.type,enabled=(self.device.type == 'cuda')):
                    outputs = self.model(images, states)
                    loss = self.criterion(outputs, actions)

                val_loss += loss.item() * images.size(0)
        return val_loss
