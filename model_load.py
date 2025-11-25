import torch
import sys
import os
CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from act.act_model import ActModel
from mlp.mlp_model import MlpModel
from diffusion.diffusion_model import DiffusionModel

def model_load(model_path,device):
    try:
        dict_data = torch.load(f"{model_path}/config.pt", map_location=device)
        if dict_data["model_type"] == "mlp":
            model_strategy = MlpModel(model_path,device)
        elif dict_data["model_type"]== "act":
            model_strategy = ActModel(model_path,device)
        elif dict_data["model_type"]== "diffusion":
            model_strategy = DiffusionModel(model_path,device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_strategy
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_history = model_load('models/mlp',device)

    dummy_image = torch.randn(3, 224, 224)  # [C, H, W]
    dummy_state = torch.randn(6)          # [D_STATE]
    
    # 3. 推論の実行
    output_actions = model_history.predict(dummy_image, dummy_state)
    
    expected_shape = torch.Size([9])
    
    if output_actions.shape == expected_shape:
        print("✅ 出力シェイプはArmPiアクション次元 (9次元) と一致しています。")
    else:
        print(f"❌ 出力シェイプが期待値 ({expected_shape}) と異なります。")
    
    print(output_actions)

    # -----
    model_history = model_load('models/act',device)

    output_actions = model_history.predict(dummy_image, dummy_state)
    
    expected_shape = torch.Size([9])
    
    if output_actions.shape == expected_shape:
        print("✅ 出力シェイプはArmPiアクション次元 (9次元) と一致しています。")
    else:
        print(f"❌ 出力シェイプが期待値 ({expected_shape}) と異なります。")
    
    print(output_actions)

    model_history = model_load('models/diffusion',device)

    output_actions = model_history.predict(dummy_image, dummy_state)
    
    expected_shape = torch.Size([9])
    
    if output_actions.shape == expected_shape:
        print("✅ 出力シェイプはArmPiアクション次元 (9次元) と一致しています。")
    else:
        print(f"❌ 出力シェイプが期待値 ({expected_shape}) と異なります。")
    
    print(output_actions)
