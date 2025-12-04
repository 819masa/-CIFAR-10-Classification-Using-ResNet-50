# CIFAR-10 Classification with ResNet50 (Optimized)

ResNet50をCIFAR-10用に最適化し、TrivialAugmentとMixup、AdamWを用いた高精度な画像分類モデルです。
最終的な精度は **94.24%** (TTA適用時) を達成しています。

## 成果物紹介資料
https://docs.google.com/presentation/d/1eCVKDGO4K4cOONah8OyigFc6Ji6w3Q-JsrtsLzt1SNQ/edit?usp=sharing

## 📂 ファイル構成

- `model.py`: モデル定義 (CIFAR-10用にStemを改造したResNet50)
- `train.py`: 学習用スクリプト (Mixup, TTA, W&Bログ対応)
- `predict.py`: 推論用スクリプト (単一画像の判定)
- `requirements.txt`: 依存ライブラリ

## 🛠 環境構築手順

Python 3.8以上を推奨します。

```bash
# ライブラリのインストール
pip install -r requirements.txt
py -m pip install torch torchvision torchaudio
```

Weights & Biases (W&B) を使用するため、初回のみログインが必要です。
```bash
wandb login
```

## 🚀 学習実行手順
以下のコマンドで学習を開始します。 学習完了後、モデルの重みが best.pt として保存されます。
```
python train.py
```

## 🖼 推論実行手順
任意の画像ファイルを分類します。

```
python predict.py --image sample.png --weights best.pt
```

出力例
------------------------------
Image: sample.png
Prediction: frog (99.82%)
------------------------------


