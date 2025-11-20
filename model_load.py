import torch
from act.act_model import ActModel
from mlp.mlp_model import MlpModel

def model_load(model_path,device):
    try:
        dict_data = torch.load(f"{model_path}/config.pt", map_location=device)
        if dict_data["model_type"] == "mlp":
            model_strategy = MlpModel(model_path,device)
        elif dict_data["model_type"]== "act":
            model_strategy = ActModel(model_path,device)
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
    
    # 4. 結果の確認
    print("\n--- 推論結果の確認 ---")
    print(f"予測アクション出力シェイプ: {output_actions.shape}")
    
    expected_shape = torch.Size([9])
    
    if output_actions.shape == expected_shape:
        print("✅ 出力シェイプはArmPiアクション次元 (9次元) と一致しています。")
    else:
        print(f"❌ 出力シェイプが期待値 ({expected_shape}) と異なります。")
    
    print(f"出力データ型: {output_actions.dtype}")

    # -----
    model_history = model_load('models/act',device)

    dummy_image = torch.randn(3, 224, 224)  # [C, H, W]
    dummy_state = torch.randn(6)          # [D_STATE]
    
    # 3. 推論の実行
    output_actions = model_history.predict(dummy_image, dummy_state)
    
    # 4. 結果の確認
    print("\n--- 推論結果の確認 ---")
    print(f"予測アクション出力シェイプ: {output_actions.shape}")
    
    expected_shape = torch.Size([9])
    
    if output_actions.shape == expected_shape:
        print("✅ 出力シェイプはArmPiアクション次元 (9次元) と一致しています。")
    else:
        print(f"❌ 出力シェイプが期待値 ({expected_shape}) と異なります。")
    
    print(f"出力データ型: {output_actions.dtype}")
