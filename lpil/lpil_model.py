from common.base_model import BaseModel 
import torch
import numpy as np
from collections import deque

class LpilModel(BaseModel):
    def __init__(self,model_path,device):
        super().__init__(model_path,device)
        torch.set_float32_matmul_precision('high')
        self.seq_len = self.model.seq_len
        self.reset()

    def reset(self):
        self.img_buffer = deque(maxlen=self.seq_len)
        self.jnt_buffer = deque(maxlen=self.seq_len)
        self.act_buffer = deque(maxlen=self.seq_len)

    def predict(self, image_tensor, state_tensor):
        with torch.no_grad():
            # (C, H, W) -> (1, C, H, W)
            curr_img = image_tensor.to(self.device).unsqueeze(0)
            # (J_dim) -> (1, J_dim)
            curr_jnt = state_tensor.to(self.device).unsqueeze(0)

            if len(self.img_buffer) == 0:
                zero_action = torch.zeros(1, 9).to(self.device)
                for _ in range(self.seq_len):
                    self.img_buffer.append(curr_img)
                    self.jnt_buffer.append(curr_jnt)
                    self.act_buffer.append(zero_action)

            past_imgs = torch.cat(list(self.img_buffer), dim=0).unsqueeze(0)
            past_jnts = torch.cat(list(self.jnt_buffer), dim=0).unsqueeze(0)
            past_acts = torch.cat(list(self.act_buffer), dim=0).unsqueeze(0)

            # BehaviorCloning.forward(past_imgs, past_jnts, past_acts, curr_img, curr_jnt)
            pred_action, _, _ = self.model(
                past_imgs, 
                past_jnts, 
                past_acts, 
                curr_img, 
                curr_jnt
            )
            pred_action = torch.clamp(pred_action, 0.0, 2.0)
            pred_action = torch.round(pred_action).to(torch.int8) - 1

            self.img_buffer.append(curr_img)
            self.jnt_buffer.append(curr_jnt)
            self.act_buffer.append(pred_action.float())

            return pred_action.squeeze(0).cpu().numpy().astype(np.int8)
