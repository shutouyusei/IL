from common.base_model import BaseModel 
import torch
import numpy as np
from collections import deque
from LPIL.bin.model_load import model_load
from LPIL.models.base.build_base import BuildModelConfig
from LPIL.models.base.train_base import TrainConfig
from LPIL.workspace.base.base_workspace import WorkSpaceConfig #
import os

class LpilModel(BaseModel):
    def __init__(self,model_path,device):
        super().__init__(model_path,device)
        torch.compile(self.model)
        torch.set_float32_matmul_precision('high')
        self.seq_len = self.model.seq_len
        self.action_dim = 9
        self.reset()

    def model_load(self, model_path,device):
        if os.path.exists(os.path.join(model_path, "config.pt")):
            config = torch.load(os.path.join(model_path, "config.pt"),weights_only=False)
            workspace_name = config['workspace_name']
            model_name = config['model_name']
            model_config = BuildModelConfig(**config['model_config'])
            workspace_config = WorkSpaceConfig(**config['workspace_config'])
        else:
            raise Exception("Config file not found. Please run train first.")
        checkpoint_path = os.path.join(model_path,"model.pt")
        self.model = model_load(workspace_name=workspace_name,model_name=model_name,workspace_config=workspace_config,model_config=model_config,checkpoint_path=checkpoint_path)
        self.model.to(device)
        self.model.eval()

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
                zero_action = torch.zeros(1, self.action_dim).to(self.device)
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
