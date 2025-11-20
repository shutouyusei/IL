from .base_model import BaseModel 
import torch
from third_party.act.detr.models.detr_vae import DETRVAE

class ActModel(BaseModel):
    def predict(self, image_tensor, state_tensor):
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            state_batch = state_tensor.unsqueeze(0).to(self.device)
            
            all_actions, _, _ = self.model(state_batch, image_batch, None)
            
            action_seq = all_actions.squeeze(0).cpu().numpy()
            
            return action_seq[0]
