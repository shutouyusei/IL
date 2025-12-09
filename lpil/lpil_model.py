from common.base_model import BaseModel 
import torch
import torch._dynamo
import numpy as np
from collections import deque
from LPIL.bin.roll_out import Rollout
from LPIL.models.base.build_base import BuildModelConfig
from LPIL.models.base.train_base import TrainConfig
from LPIL.workspace.base.base_workspace import WorkSpaceConfig
import os

class LpilModel(BaseModel):
    def __init__(self, model_path, device,task_path):
        super().__init__(model_path, device,task_path)
        torch._dynamo.config.suppress_errors = True
        torch.set_float32_matmul_precision('high')

    def model_load(self, model_path, device,task_path):
        if os.path.exists(os.path.join(model_path, "config.pt")):
            config = torch.load(os.path.join(model_path, "config.pt"), weights_only=False)
            workspace_name = config['workspace_name']
            model_name = config['model_name']
            model_config = BuildModelConfig(**config['model_config'])
            workspace_config = WorkSpaceConfig(**config['workspace_config'])
        else:
            raise Exception("Config file not found. Please run train first.")

        checkpoint_path = os.path.join(model_path, "model.pt")
        model_config.checkpoint_path = checkpoint_path

        task_path = f"{task_path}/goal_latent.pth"
        # Rollout (モデルを含むラッパー) の初期化
        self.roll_out = Rollout(
            workspace_name=workspace_name,
            model_name=model_name,
            workspace_config=workspace_config,
            model_config=model_config,
            task_path=task_path
        )

    def predict(self, image_tensor, state_tensor):
        return self.roll_out.predict(image_tensor, state_tensor)
