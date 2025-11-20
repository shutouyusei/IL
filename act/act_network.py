import torch
import torch.nn as nn
from third_party.act.detr.models import build_ACT_model
from third_party.act.detr.models.detr_vae import DETRVAE
from types import SimpleNamespace
import torch

def build_ACT(state_dim,action_dim):
    args = SimpleNamespace(
        # -- DETR
        masks=False, 
        dilation=False,
        pre_norm=True,  #  Pre-Normalizationを有効化
        aux_loss = False,
        # --- ArmPi I/O 特化パラメータ ---
        state_dim=state_dim,            # ロボットの状態次元: ArmPiの全関節位置 (例: 6DoF)
        action_dim=action_dim,           # アクションの次元: ロボットに送るコマンドの総数 (例: 9つの値)
        camera_names=['primary'], # 使用するカメラのリスト (単一カメラを想定)
        
        # --- Transformer / DETR アーキテクチャ設定 ---
        hidden_dim=256,         # トランスフォーマーの隠れ層の次元数 (標準的な設定)
        enc_layers=4,           # エンコーダーのレイヤー数
        dec_layers=6,           # デコーダーのレイヤー数 (通常エンコーダーより多い)
        dim_feedforward=1024,   # FFN (Feed-Forward Network) の内部次元
        dropout=0.1,            # ドロップアウト率
        nheads=8,               # マルチヘッドアテンションのヘッド数
        num_queries=100,        # 予測するアクションチャンクの長さ (例: 10Hzで10秒間の行動)
        
        # --- バックボーン設定 (画像特徴抽出) ---
        backbone='resnet18',    # 画像特徴抽出に使うCNN (ResNet-18は高速でデバッグ向き)
        position_embedding='sine', # 位置エンコーディングのタイプ ('sine'または'learned')
        lr_backbone=1e-5,       # バックボーンの学習率 (ファインチューニングを想定し低めに設定)

        # --- 学習・実行環境設定 ---
        kl_weight=1e-5,         # VAE (Variational AutoEncoder) のKL損失の重み (Actで行動多様性を高めるため)
        device='cuda' if torch.cuda.is_available() else 'cpu', # 実行デバイスの自動設定
    )
    return  build_ACT_model(args)
