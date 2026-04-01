[English](README.md) | [日本語](README.ja.md)

# IL: 模倣学習アルゴリズムライブラリ

[Hiwonder ArmPi Pro](https://www.hiwonder.com/) ロボットアームの操作を対象とした、4種類の模倣学習アルゴリズムを統一インターフェースで実装したライブラリです。

## 実装済みアルゴリズム

| アルゴリズム | 概要 | 主要アーキテクチャ |
|-------------|------|-------------------|
| **ACT** (Action Chunking with Transformers) | Transformerベースの行動系列予測 + VAEによるマルチモーダル行動生成 | DETR + ResNet-18 + VAE |
| **Diffusion Policy** | 拡散モデルによる方策生成 | DDPM/DDIM + CNN/Transformer |
| **MLP** (ベースライン) | 単純な全結合ネットワークによるシングルステップ予測 | ResNet-18 + MLP融合ヘッド |
| **LPIL** (Latent Policy Imitation Learning) | ゴール条件付き潜在表現学習 | 外部LPILフレームワーク |

## アーキテクチャ

```
┌─────────────────────────────────────────────┐
│         ユーザーインターフェース               │
│   train_main.py（学習）                       │
│   model_load.py（推論、ファクトリパターン）     │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
     学習             推論
       │                │
   ┌───┴───┬────┬──────┐
   ▼       ▼    ▼      ▼
  MLP    ACT  Diff   LPIL
   └───┬───┴────┴──────┘
       ▼
  common/
  ├── base_model.py    # 推論用抽象基底クラス
  ├── trainer.py       # 早期終了・チェックポイント付き基底トレーナー
  ├── armpi_const.py   # ロボットの行動/状態定義
  └── read_hdf.py      # HDF5データセットリーダー
```

## リポジトリ構成

```
il/
├── model_load.py          # ファクトリ：モデルタイプに応じた読み込み
├── train_main.py          # 統一学習エントリポイント（CLI）
├── act/                   # ACT アルゴリズム
│   ├── act_model.py       #   推論ラッパー
│   ├── act_network.py     #   ネットワーク構造（DETR + VAE）
│   ├── act_trainer.py     #   KLダイバージェンス損失による学習
│   └── act_armpi_dataset.py  # データセット：100ステップの行動チャンク
├── diffusion/             # Diffusion Policy アルゴリズム
│   ├── diffusion_model.py       # DDPM/DDIMスケジューラによる推論
│   ├── diffusion_network.py     # Diffusion Policyビルダー
│   ├── diffusion_trainer.py     # ノイズスケジューラによる学習
│   └── diffusion_armpi_dataset.py  # データセット：ホライゾンベース系列
├── mlp/                   # MLP ベースライン
│   ├── mlp_model.py       #   推論ラッパー
│   ├── mlp_network.py     #   ResNet18 + MLP融合
│   ├── mlp_trainer.py     #   クロスエントロピー学習
│   └── mlp_armpi_dataset.py  # データセット：シングルステップ
├── lpil/                  # LPIL アルゴリズム
│   ├── lpil_model.py      #   ゴール潜在変数による推論
│   └── lpil_model_convert.py  # 学習結果の変換
├── common/                # 共有ユーティリティ
│   ├── base_model.py      #   全モデル共通の抽象基底クラス
│   ├── trainer.py         #   基底トレーナー（早期終了、チェックポイント）
│   ├── armpi_const.py     #   ロボット定数（9行動、6状態）
│   └── read_hdf.py        #   HDF5データセットリーダー
└── third_party/           # 外部実装（gitサブモジュール）
    ├── act/               #   ACT（DETRベース）
    ├── diffusion/         #   Diffusion Policy
    └── LPIL/              #   LPILフレームワーク
```

## クイックスタート

### 前提条件

```bash
# Conda環境の作成
conda env create -f environment.yml
conda activate armpi_env

# サブモジュールの初期化
git submodule update --init --recursive
```

### 学習

```bash
# MLPベースラインの学習
python train_main.py --file_name mlp_experiment --task_name bring_up_test --model mlp --epochs 50

# ACTの学習
python train_main.py --file_name act_experiment --task_name bring_up_test --model act --epochs 100

# Diffusion Policyの学習
python train_main.py --file_name diffusion_experiment --task_name bring_up_test --model diffusion --epochs 100
```

**学習引数：**

| 引数 | 説明 | デフォルト |
|------|------|------------|
| `--file_name` | モデル保存名 | （必須） |
| `--task_name` | `datasets/` 内のフォルダ名 | （必須） |
| `--model` | アルゴリズム: `mlp`, `act`, `diffusion` | （必須） |
| `--batch_size` | バッチサイズ | 32 |
| `--val_split` | 検証データ分割比率 | 0.2 |
| `--epochs` | エポック数 | 10 |
| `--learning_rate` | 学習率 | 0.001 |
| `--early_stop_patience` | 早期終了の猶予エポック数 | 10 |

### 推論

```python
import torch
from model_load import model_load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_load('models/act_experiment', device)

# 入力: カメラ画像 [3, 224, 224] + 関節状態 [6]
image = torch.randn(3, 224, 224)
state = torch.randn(6)

# 出力: 9次元の行動ベクトル
action = model.predict(image, state)  # -> torch.Size([9])
```

## データフォーマット

学習データは `datasets/<task_name>/` にHDF5ファイルとして保存されます。

各 `.h5` ファイルは `sync_data` キーに、同期済みのpandas DataFrameを含みます：
- **画像**: カメラフレーム（RGB、ファイルパスとして保存）
- **状態**: 6つの関節位置 (`joint1_pos` ... `joint5_pos`, `r_joint_pos`)
- **行動**: 9次元のコマンド：

| インデックス | 行動 | 説明 |
|-------------|------|------|
| 0 | `chassis_move_forward` | 前進/後退 |
| 1 | `chassis_move_right` | 左右移動 |
| 2 | `angular_right` | 回転 |
| 3 | `arm_x` | アームX位置 |
| 4 | `arm_y` | アームY位置 |
| 5 | `arm_z` | アームZ位置 |
| 6 | `arm_alpha` | ロール姿勢 |
| 7 | `rotation` | 手首回転 |
| 8 | `gripper_close` | グリッパー開閉 |

## 連携

このライブラリは [ArmPi](https://github.com/shutouyusei/ArmPi) のgitサブモジュールとして、実機ロボットへのデプロイに使用されています。ArmPiでは `ros/myapp/ai_model_service/src/ai_modules/` にマウントされ、ROS推論サービスノードから呼び出されます。

## ライセンス

本プロジェクトは研究・教育目的です。
