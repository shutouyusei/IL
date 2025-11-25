import robomimic.config.config as rcfg

_orig_setitem = rcfg.Config.__setitem__

def _safe_setitem(self, key, value):
    try:
        object.__getattribute__(self, "__parent")
        object.__getattribute__(self, "__key")
        object.__getattribute__(self, "__all_locked")
        object.__getattribute__(self, "__key_locked")
    except AttributeError:
        object.__setattr__(self, "__parent", None)
        object.__setattr__(self, "__key", None)
        object.__setattr__(self, "__all_locked", None)
        object.__setattr__(self, "__key_locked", None)
    
    return _orig_setitem(self, key, value)

rcfg.Config.__setitem__ = _safe_setitem

from common.base_model import BaseModel 
import torch
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionModel(BaseModel):
    def predict(self, image_tensor, state_tensor):
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            state_batch = state_tensor.unsqueeze(0).to(self.device)

            if image_batch.shape[1] == 1:
                image_batch = image_batch.repeat(1, 3, 1, 1)
            n_obs_steps = getattr(self.model, 'n_obs_steps', 2)

            image_batch = image_batch.unsqueeze(1).repeat(1, n_obs_steps, 1, 1, 1)
            state_batch = state_batch.unsqueeze(1).repeat(1, n_obs_steps, 1)

            batch = {
                'state': state_batch,
                'primary': image_batch
            }

            result = self.model.predict_action(batch)
            
            all_actions = result['action']
            
            action = all_actions[0, 0, :].cpu().numpy()
            action = np.round(action).astype(np.int8)

            return action
