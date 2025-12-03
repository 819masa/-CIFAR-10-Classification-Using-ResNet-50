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

# model.py からモデル定義関数を読み込む
# ※ 同じフォルダに model.py がある前提です
from model import get_model 

# --- 1. 設定とハイパーパラメータ ---
hyperparameters = {
    "project_name": "clasification of cifar-10 by ResNet50",
    "experiment_name": "ResNet50_Stem_TrivialAug_AdamW_50ep_TTA_RE", 
    "note": "AdamW (lr=0.001, wd=1e-2). Modified Stem + TrivialAugment + RandomErasing. No Mixup.",
    "architecture": "ResNet50_CIFAR_Optimized",
    "dataset": "CIFAR-10",
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 1e-2,
    "momentum": 0.9,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "seed": 42,
    "resize": 32,
    "use_mixup": False,  # 今回はOFF
    "mixup_alpha": 1.0,
    "mixup_epochs": 0,
}

# --- 2. シード固定関数 ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# --- 3. 学習関数 ---
def train_one_epoch(epoch, model, loader, optimizer, criterion, device, config):
    model.train()
    sum_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixup (今回はOFF設定ですがロジックは残しておきます)
        if config.use_mixup and epoch <= config.mixup_epochs:
            lam = np.random.beta(config.mixup_alpha, config.mixup_alpha)
            batch_size = images.size(0)
            index = torch.randperm(batch_size).to(device)
            mixed_images = lam * images + (1 - lam) * images[index]
            outputs = model(mixed_images)
            loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
            mode = "Mixup"
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            mode = "Normal"

        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = sum_loss / len(loader)
    acc = correct / total
    current_lr = optimizer.param_groups[0]['lr']

    print(f"[Train] Epoch {epoch} ({mode}): Loss={avg_loss:.4f}, Acc={acc:.4f}, LR={current_lr:.6f}")
    
    wandb.log({
        "epoch": epoch,
        "train/loss": avg_loss,
        "train/accuracy": acc,
        "train/learning_rate": current_lr,
        "mixup_mode": 1 if mode == "Mixup" else 0
    })

# --- 4. 評価関数 (通常) ---
def evaluate(epoch, model, loader, criterion, device, config, classes, log_results=True):
    model.eval()
    sum_loss = 0.0
    all_preds = []
    all_labels = []
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

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 誤分類収集
            if len(misclassified_images) < 32:
                mask = predicted != labels_dev
                if mask.any():
                    wrong_imgs = images_dev[mask].cpu()
                    wrong_preds = predicted[mask].cpu()
                    wrong_labels = labels_dev[mask].cpu()
                    for img, p, l in zip(wrong_imgs, wrong_preds, wrong_labels):
                        if len(misclassified_images) < 32:
                            misclassified_images.append(img)
                            misclassified_preds.append(p.item())
                            misclassified_labels.append(l.item())

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
        
        # 最終エポックのみ画像ログなどを送信
        if epoch == config.epochs:
            # 1. 混同行列
            log_dict["val/confusion_matrix"] = wandb.plot.confusion_matrix(
                probs=None, y_true=all_labels, preds=all_preds, class_names=classes
            )
            # 2. 誤分類サンプル (逆正規化して表示)
            mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
            std = torch.tensor((0.2023, 0.1994, 0.2010)).view(3, 1, 1)
            wandb_images = []
            for img, p, l in zip(misclassified_images, misclassified_preds, misclassified_labels):
                img = img * std + mean # Un-normalize
                img = torch.clamp(img, 0, 1)
                caption = f"True: {classes[l]} / Pred: {classes[p]}"
                wandb_images.append(wandb.Image(img, caption=caption))
            
            log_dict["val/misclassified_examples"] = wandb_images

        wandb.log(log_dict)
    return acc

# --- 5. TTA評価関数 ---
def evaluate_with_tta(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\n🚀 Starting TTA Evaluation (Original + Horizontal Flip)...")
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # 1. Original
            outputs1 = model(images)
            probs1 = F.softmax(outputs1, dim=1)

            # 2. Flipped
            images_flipped = torch.flip(images, dims=[3])
            outputs2 = model(images_flipped)
            probs2 = F.softmax(outputs2, dim=1)

            # Average
            avg_probs = (probs1 + probs2) / 2.0
            _, predicted = torch.max(avg_probs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"--------------------------------------------------")
    print(f"✅ TTA Result:")
    print(f"   Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"   F1 Score : {f1:.4f}")
    print(f"--------------------------------------------------")
    
    wandb.log({
        "test/tta_accuracy": acc,
        "test/tta_precision": precision,
        "test/tta_recall": recall,
        "test/tta_f1_score": f1
    })
    return acc

# --- 6. メイン実行部 ---
def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # W&B Init
    wandb.init(
        project=hyperparameters["project_name"],
        name=hyperparameters["experiment_name"],
        config=hyperparameters
    )
    config = wandb.config # config.seed のようにアクセス可能にする
    
    set_seed(config.seed)
    
    # Data Augmentation (RandomErasingを追加！)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5), # 実験名に合わせて追加しました
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Dataset & Loader
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model Setup
    model = get_model(device) # model.pyから取得
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Training Loop
    print("Training Started...")
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_one_epoch(epoch, model, train_loader, optimizer, criterion, device, config)
        evaluate(epoch, model, test_loader, criterion, device, config, classes)
        scheduler.step()

    print(f"Total Training Time: {time.time() - start_time:.2f}s")

    # Save & TTA
    save_path = "best.pt"
    torch.save(model.state_dict(), save_path)
    
    # モデルをロードし直してTTA評価
    model.load_state_dict(torch.load(save_path))
    evaluate_with_tta(model, test_loader, device)

    wandb.finish()

if __name__ == "__main__":
    main()

