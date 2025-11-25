import torch
import torch.nn as nn
from types import SimpleNamespace
import copy

# 必要なライブラリのインポート (diffusion_policyがパスに通っている前提)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

def build_diffusion_policy_model(args):
    """
    argsの内容に基づいてDiffusion Policyモデルを構築する実体関数
    """
    # 1. Shape Metaの構築 (入力データの形状定義)
    # ----------------------------------------------------------
    shape_meta = {
        "obs": {
            "state": {
                "shape": [args.state_dim],
                "type": "low_dim"
            }
        },
        "action": {"shape": [args.action_dim]},
    }
    
    # 画像入力の定義
    for camera_name in args.camera_names:
        shape_meta["obs"][camera_name] = {
            "shape": [3, args.image_size[1], args.image_size[0]], # C, H, W
            "type": "rgb",
        }
    # 2. Noise Schedulerの構築
    # ----------------------------------------------------------
    noise_scheduler_args = {
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "clip_sample": True,
        "prediction_type": "epsilon",
        "num_train_timesteps": 100, # DDPMのデフォルト
    }

    if args.scheduler == "ddpm":
        noise_scheduler_args.update({
            "variance_type": "fixed_small"
        })
        noise_scheduler = DDPMScheduler(**noise_scheduler_args)
    
    elif args.scheduler == "ddim":
        noise_scheduler_args.update({
            "num_train_timesteps": 100, # 学習時のステップ数
            "set_alpha_to_one": True,
            "steps_offset": 0,
        })
        noise_scheduler = DDIMScheduler(**noise_scheduler_args)
    else:
        raise ValueError(f"Invalid scheduler: {args.scheduler}")

    # 3. Policy (Model) の構築
    # ----------------------------------------------------------
    if args.backbone == "cnn":
        # CNN (ResNet + UNet) 構成
        policy_args = {
            "shape_meta": shape_meta,
            "noise_scheduler": noise_scheduler,
            "horizon": args.horizon,
            "n_action_steps": args.n_action_steps,
            "n_obs_steps": args.n_obs_steps,
            "num_inference_steps": args.num_inference_steps, # 推論時のステップ数
            "crop_shape": args.image_crop_size[::-1], # (H, W)
            "obs_encoder_group_norm": True,
            "eval_fixed_crop": True,
            # CNN固有パラメータ
            "down_dims": [512, 1024, 2048],
            "kernel_size": 5,
            "n_groups": 8,
            "cond_predict_scale": True,
            "obs_as_global_cond": True,
            "diffusion_step_embed_dim": 128,
        }
        model = DiffusionUnetHybridImagePolicy(**policy_args)

    elif args.backbone == "transformer":
        # Transformer (DiT-like) 構成
        policy_args = {
            "shape_meta": shape_meta,
            "noise_scheduler": noise_scheduler,
            "horizon": args.horizon,
            "n_action_steps": args.n_action_steps,
            "n_obs_steps": args.n_obs_steps,
            "num_inference_steps": args.num_inference_steps,
            "crop_shape": args.image_crop_size[::-1], # (H, W)
            "obs_encoder_group_norm": True,
            "eval_fixed_crop": True,
            # Transformer固有パラメータ
            "n_layer": 8,
            "n_cond_layers": 0,
            "n_head": 4,
            "n_emb": 256,
            "p_drop_emb": 0.0,
            "p_drop_attn": 0.3,
            "causal_attn": True,
            "time_as_cond": True,
            "obs_as_cond": True,
        }
        model = DiffusionTransformerHybridImagePolicy(**policy_args)
    else:
        raise ValueError(f"Invalid backbone: {args.backbone}")

    # デバイスへの転送
    device = torch.device(args.device)
    model.to(device)
    
    return model


def build_diffusion_policy(state_dim, action_dim):
    """
    Diffusion Policyモデル設定を定義し、モデルを構築して返す関数
    """
    args = SimpleNamespace(
        # --- 基本設定 ---
        state_dim=state_dim,
        action_dim=action_dim,
        camera_names=['primary'], # データセットに合わせて変更してください
        device='cuda' if torch.cuda.is_available() else 'cpu',

        # --- モデルアーキテクチャ設定 ---
        # backbone: 'cnn' (ResNet18 + UNet) or 'transformer' (MinGPT)
        backbone='cnn', 
        
        # scheduler: 'ddpm' (高精度/遅い) or 'ddim' (高速)
        # 注意: TransformerバックボーンはDDIMと相性が悪い場合があります(参照コードより)
        scheduler='ddpm', 

        # --- シーケンス設定 (Horizon) ---
        # horizon: 予測する未来のステップ数 (入力履歴 + 出力未来)
        # cnn推奨: 16, transformer推奨: 10
        horizon=16, 
        
        # n_obs_steps: 入力として使う過去の観測ステップ数 (通常 2)
        n_obs_steps=2,
        
        # n_action_steps: 一度の推論で実行するアクションステップ数 (通常 8)
        # Receding Horizon Controlのために horizon より小さく設定する
        n_action_steps=8,

        # --- 画像処理設定 ---
        # image_size: リサイズ後のサイズ [W, H]
        image_size=[320, 240],
        # image_crop_size: ランダムクロップ後のサイズ [W, H]
        image_crop_size=[224, 224],

        # --- 推論設定 ---
        # num_inference_steps: 拡散過程のデノイズステップ数
        # DDPMなら100, DDIMなら少なめ(例: 8~16)でも動作可能
        num_inference_steps=100, 
    )

    # Transformer選択時のデフォルト設定の上書き（参照コードのロジックに準拠）
    if args.backbone == "transformer":
        args.horizon = 10
        if args.scheduler == "ddim":
            print("Warning: Transformer backbone might not work well with DDIM.")

    # DDIM選択時の推論ステップ数調整
    if args.scheduler == "ddim":
        args.num_inference_steps = 8

    return build_diffusion_policy_model(args)
