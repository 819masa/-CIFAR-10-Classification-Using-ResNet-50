import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import torch.nn.functional as F

# CIFAR-10 クラス定義
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. モデル準備
    model = get_model(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: 重みファイル '{weights_path}' が見つかりません。")
        return

    model.eval()

    # 2. 画像前処理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error: 画像の読み込みに失敗しました。 {e}")
        return

    # 3. 推論実行
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        
    # 4. 結果表示
    pred_idx = torch.argmax(probabilities).item()
    pred_class = CLASSES[pred_idx]
    confidence = probabilities[pred_idx].item()

    print("-" * 30)
    print(f"Image: {image_path}")
    print(f"Prediction: {pred_class} ({confidence*100:.2f}%)")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR-10 Inference CLI')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--weights', type=str, default='best.pt', help='Path to the trained weights')
    
    args = parser.parse_args()
    predict(args.image, args.weights)
