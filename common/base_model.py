import torch
import numpy as np

class BaseModel:
    def __init__(self, model_path,device):
        self.device = device
        model_data = torch.load(f"{model_path}/model.pt",map_location=self.device,weights_only=False)
        self.model = model_data['model']
        self.model.to(device)
        self.model.eval()

    def predict(self, image_tensor, state_tensor):
        raise NotImplementedError
