import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import random
import time
import wandb
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import get_model

hyperparameters = {
    "project_name": "clasification of cifar-10 by ResNet50",
    "experiment_name": "ResNet50_Stem_TrivialAug_AdamW_50ep_TTA_ramdomerasing", # 実験名
    "note": "AdamW (lr=0.001, wd=1e-2). Modified Stem + TrivialAugment.no mixup and add TTA,ramdomerasing.", # 施策メモ
    "architecture": "ResNet50_CIFAR_Optimized",
    "dataset": "CIFAR-10",
    "epochs": 50,              # 動作確認のため少なめに設定しています。
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 1e-2,      # AdamW用に変更
    "momentum": 0.9,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR", # 学習率スケジューラを追加
    "seed": 42,
    "resize": 32,              # CIFAR-10のデフォルト
    "use_mixup": False,       # Mixupを使うかスイッチ
    "mixup_alpha": 1.0,      # 混ぜ具合のパラメータ（1.0が標準）
    "mixup_epochs": 0,      # 「最初の40エポックだけ」Mixupする（残りの10は普通に学習）
}



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(config.seed)



# --- 6. 学習・評価関数の定義 ---


def train_one_epoch(epoch, model, loader, optimizer, criterion, device, config): # configを受け取るように変更
    model.train()
    sum_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # === Mixup の判定 ===
        # 設定がON、かつ 指定エポック以内なら Mixup を実行
        if config.use_mixup and epoch <= config.mixup_epochs:

            # 1. Beta分布から混ぜる比率 (lambda) を決める
            # alpha=1.0 なら 0~1 の間で均等に選ばれる
            lam = np.random.beta(config.mixup_alpha, config.mixup_alpha)

            # 2. バッチ内の画像をシャッフルするためのインデックスを作る
            batch_size = images.size(0)
            index = torch.randperm(batch_size).to(device)

            # 3. 画像を混ぜる！ ( mixed_x = λx + (1-λ)x_shuffle )
            mixed_images = lam * images + (1 - lam) * images[index]

            # 4. モデルに通す
            outputs = model(mixed_images)

            # 5. Lossを混ぜる！ ( Loss = λ * Loss(y1) + (1-λ) * Loss(y2) )
            # ラベル自体を混ぜるのではなく、それぞれの正解に対するLossを計算して混ぜます
            loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])

        else:
            # === 通常学習 (後半エポック または Mixup OFF時) ===
            outputs = model(images)
            loss = criterion(outputs, labels)

        # --- 以下は共通 ---
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        # 精度の計算（Mixup中は正確な正解率が出ないので、主ラベルで近似計算）
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = sum_loss / len(loader)
    acc = correct / total
    current_lr = optimizer.param_groups[0]['lr']

    # ログ表示（Mixup中かどうか分かるようにする）
    mode = "Mixup" if (config.use_mixup and epoch <= config.mixup_epochs) else "Normal"
    print(f"[Train] Epoch {epoch} ({mode}): Loss={avg_loss:.4f}, Acc={acc:.4f}, LR={current_lr:.6f}")

    wandb.log({
        "epoch": epoch,
        "train/loss": avg_loss,
        "train/accuracy": acc,
        "train/learning_rate": current_lr,
        "mixup_mode": 1 if mode == "Mixup" else 0 # グラフで切り替わりが見えるように
    })



def evaluate(epoch, model, loader, criterion, device, log_results=True):
    model.eval()
    sum_loss = 0.0
    all_preds = []
    all_labels = []
    # 誤分類画像を保存するためのリスト
    misclassified_images = []
    misclassified_preds = []
    misclassified_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images_dev = images.to(device)
            labels_dev = labels.to(device)

            outputs = model(images_dev)
            loss = criterion(outputs, labels_dev)
            sum_loss += loss.item()

            _, predicted = outputs.max(1)

            # データをCPUに戻してリストに追加
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # ---  誤分類サンプルの収集 ---
            if len(misclassified_images) < 32:
                mask = predicted != labels_dev
                if mask.any():
                    # 修正: images（CPU）ではなく images_dev（GPU）を使う
                    # GPU同士で計算してから .cpu() で戻すことでエラーを回避
                    wrong_imgs = images_dev[mask].cpu()
                    wrong_preds = predicted[mask].cpu()
                    wrong_labels = labels_dev[mask].cpu()

                    for img, p, l in zip(wrong_imgs, wrong_preds, wrong_labels):
                        if len(misclassified_images) < 32:
                            misclassified_images.append(img)
                            misclassified_preds.append(p.item())
                            misclassified_labels.append(l.item())

    # 指標計算
    avg_loss = sum_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"[Val] Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    if log_results:
        log_dict = {
            "epoch": epoch,
            "val/loss": avg_loss,
            "val/accuracy": acc,
            "val/precision": precision,
            "val/recall": recall,
            "val/f1_score": f1,
        }

        # 最終エポックのみ詳細なアーティファクトを記録
        if epoch == config.epochs:
            # 1. 混同行列
            log_dict["val/confusion_matrix"] = wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=classes
            )

            # 2. 誤分類サンプルの画像記録
            wandb_images = []
            for img, p, l in zip(misclassified_images, misclassified_preds, misclassified_labels):
                img = torch.clamp(img, 0, 1)
                caption = f"True: {classes[l]} / Pred: {classes[p]}"
                wandb_images.append(wandb.Image(img, caption=caption))

            log_dict["val/misclassified_examples"] = wandb_images

        wandb.log(log_dict)

    return acc

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 詳細指標対応版 TTA評価関数 ---
def evaluate_with_tta(model, loader, device):
    model.eval()
    
    # 全データの予測と正解を貯めるリスト
    all_preds = []
    all_labels = []
    
    print("\n🚀 Starting TTA Evaluation (Original + Horizontal Flip)...")
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # 1. そのまま予測
            outputs1 = model(images)
            probs1 = F.softmax(outputs1, dim=1)

            # 2. 左右反転して予測
            images_flipped = torch.flip(images, dims=[3])
            outputs2 = model(images_flipped)
            probs2 = F.softmax(outputs2, dim=1)

            # 3. アンサンブル (平均)
            avg_probs = (probs1 + probs2) / 2.0
            
            # 予測ラベルを取得
            _, predicted = torch.max(avg_probs.data, 1)
            
            # リストに追加 (CPUに戻してnumpy化)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 指標の計算 (Macro平均) ---
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"--------------------------------------------------")
    print(f"✅ TTA Result:")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"--------------------------------------------------")
    
    # W&Bに記録 (通常のvalと区別するために tta/ をつける)
    wandb.log({
        "test/tta_accuracy": acc,
        "test/tta_precision": precision,
        "test/tta_recall": recall,
        "test/tta_f1_score": f1
    })
    
    return acc



