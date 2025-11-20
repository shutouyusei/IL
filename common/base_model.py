import torch
import numpy as np

class BaseModel:
    def __init__(self, device,model_state_dict):
        self.device = device
        self.model = self.__load_model(model_state_dict)

    def __load_model(sefl.model_state_dict):
        raise NotImplementedError

    def predict(self, image_tensor, state_tensor):
        raise NotImplementedError
