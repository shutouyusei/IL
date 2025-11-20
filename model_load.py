import torch
from .act.act_model import ActModel
from .mlp.mlp_model import MlpModel

def model_load(model_path,device):
    try:
        dict_data = torch.load(model_path, map_location=device)
        if dict_data["model_type"] == "mlp":
            model_strategy = MlpModel(dict_data["model_state_dict"], device)
        elif dict_data["model_type"]== "act":
            model_strategy = ActModel(dict_data["model_state_dict"], device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_strategy
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

