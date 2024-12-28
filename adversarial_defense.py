import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
import numpy as np

# 配置日志记录
logging.basicConfig(filename='accuracy_defense.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 加载预训练的ResNet50模型
model = models.resnet50(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型移动到设备上
model.eval()  # 设置为评估模式

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载ImageNet验证集
val_dataset = datasets.ImageFolder(root='fgsm_images', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


def add_random_noise(images, noise_factor=0.1):
    """Add random Gaussian noise to the images."""
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1)


# 定义函数来计算准确率，包括对抗防御
def calculate_accuracy_with_defense(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)  # 将数据移动到设备上

            # Apply defense by adding random noise
            defended_data = add_random_noise(data)

            output = model(defended_data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            print(f"batch {i + 1}, Accuracy: {correct / total * 100:.2f}%")
            logging.info(f"batch {i + 1}, Accuracy: {correct / total * 100:.2f}%")
    return correct / total


# 计算带防御的准确率
accuracy = calculate_accuracy_with_defense(model, val_loader)
print(f"ResNet50在对抗样本上的防御后准确率为: {accuracy * 100:.2f}%")
logging.info(f"ResNet50在对抗样本上的防御后准确率为: {accuracy * 100:.2f}%")