# CIFAR-10 クラス定義
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. モデル準備
    model = get_model(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: 重みファイル '{weights_path}' が見つかりません。学習を実行してください。")
        return

    model.eval()


# --- 4. データ準備 ---
# 将来的にAugmentationを変える場合はここをいじる
transform_train = transforms.Compose([
    # 1. 基本の幾何学変換 
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # 2. ここをTrivialAugmentに
    transforms.TrivialAugmentWide(),
    # 3. テンソル化と正規化
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # (オプション) 
    #transforms.RandomErasing(p=0.5)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=2)



start_time = time()


for epoch in range(1, config.epochs + 1):
    # 引数に config を追加
    train_one_epoch(epoch, model, train_loader, optimizer, criterion, device, config)
    evaluate(epoch, model, test_loader, criterion, device)
    scheduler.step()

print(f"Total Training Time: {time() - start_time:.2f}s")

# --- 8. モデルの保存とアーティファクト記録 ---
save_path = "resnet50_cifar10.pth"
torch.save(model.state_dict(), save_path)

# W&Bにモデルファイルをアップロード
artifact = wandb.Artifact('model-resnet50', type='model')
artifact.add_file(save_path)
run.log_artifact(artifact)

print("Training Finished and Artifacts logged.")

model.load_state_dict(torch.load(save_path))

# 学習完了後のモデルを使ってTTA評価を実行
tta_acc = evaluate_with_tta(model, test_loader, device)
print(f"Final Result with TTA: tta_acc")

wandb.finish()
