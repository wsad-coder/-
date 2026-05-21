import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 48
SAVE_MODEL_PATH = "emotion_model.pth"
EMOTION_LABELS = {0: "生气", 1: "厌恶", 2: "恐惧", 3: "开心", 4: "难过", 5: "惊讶", 6: "中性"}

# 图像预处理（与训练时保持一致）
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 数据集类
class FERDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_info = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        label = int(self.data_info.iloc[index, 0])
        pixel_str = self.data_info.iloc[index, 1]
        pixel_arr = np.array(pixel_str.split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixel_arr).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label

# 模型定义（与训练时保持一致）
class EmotionCNN(nn.Module):
    def __init__(self, num_class=7):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        out = self.fc_layer(x)
        return out

# 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    total_confidence = 0.0
    confidence_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
            # 计算平均置信度
            max_probs, _ = torch.max(probs, dim=1)
            total_confidence += max_probs.sum().item()
            confidence_list.extend(max_probs.cpu().numpy())
    
    accuracy = correct / total * 100
    avg_confidence = total_confidence / total * 100
    
    print(f"测试集准确率: {accuracy:.2f}%")
    print(f"平均置信度: {avg_confidence:.2f}%")
    print(f"最低置信度: {min(confidence_list)*100:.2f}%")
    print(f"最高置信度: {max(confidence_list)*100:.2f}%")
    
    return accuracy, avg_confidence, confidence_list

# 主函数
if __name__ == "__main__":
    print("加载测试数据集...")
    test_dataset = FERDataset("test.csv", transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"测试集大小: {len(test_dataset)}")
    
    print("\n加载模型...")
    model = EmotionCNN().to(device)
    try:
        model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()
    
    print("\n开始评估...")
    evaluate_model(model, test_loader)
