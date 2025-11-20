from common.base_model import BaseModel 
from .mlp_network import MlpNetwork
import torch
from common.armpi_const import *
import numpy as np

class MlpModel(BaseModel):
    def predict(self, image_tensor, state_tensor):
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            state_batch = state_tensor.unsqueeze(0).to(self.device)

            logits = self.model(state_batch, image_batch)
            
            predicted_indicies = torch.argmax(logits, dim=1)
            predicted_actions = (predicted_indicies - 1).cpu().numpy().squeeze()
            return predicted_actions.astype(np.int8)
