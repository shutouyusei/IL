from LPIL.bin.model_load import model_load
from LPIL.models.base.build_base import BuildModelConfig
from LPIL.models.base.train_base import TrainConfig
from LPIL.workspace.base.base_workspace import WorkSpaceConfig #
import argparse
import os
import torch
import argparse
from pathlib import Path
import dataclasses

def get_parser():
    parser = argparse.ArgumentParser(description="Train a workspace model",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('save_folder_name',type=str,help='Path to the save folder')
    parser.add_argument('checkpoint_name',type=str,help='Path to the checkpoint')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    #check exists config file
    if os.path.exists(os.path.join("results", args.save_folder_name, "config.pth")):
        config = torch.load(os.path.join("results", args.save_folder_name, "config.pth"),weights_only=False)
        workspace_name = config['workspace_name']
        model_name = config['model_name']
        model_config = BuildModelConfig(**config['model_config'])
        model_config.checkpoint_path = os.path.join("results", args.save_folder_name, args.checkpoint_name)
        workspace_config = WorkSpaceConfig(**config['workspace_config'])
    else:
        raise Exception("Config file not found. Please run train first.")

    # =========================
    model = model_load(workspace_name=workspace_name,model_name=model_name,workspace_config=workspace_config,model_config=model_config)

    Path(f"models/lpil_foundation").mkdir(parents=True, exist_ok=True)
    config_data = { 
        'model_type': "lpil",
        'workspace_name': workspace_name,
        'model_name': model_name,
        'model_config': dataclasses.asdict(model_config),
        'workspace_config': dataclasses.asdict(workspace_config),
    }
    torch.save(config_data,f"models/lpil_foundation/config.pt")
    save_data = { 'model_state_dict': model.state_dict() }
    torch.save(save_data, f"models/lpil_foundation/model.pt")
