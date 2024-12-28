import torch
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import logging

# 配置日志记录
logging.basicConfig(filename='fgsm_attack.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 加载预训练的ResNet50模型
model = models.resnet50(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型移动到设备上
model.eval()  # 设置为评估模式

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载ImageNet验证集
val_dataset = datasets.ImageFolder(root='ILSVRC2012_img_val_categories', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 创建输出目录以保存对抗样本
output_dir = 'path_to_save_fgsm_images'
os.makedirs(output_dir, exist_ok=True)


def save_image(tensor, path):
    """Helper function to save a tensor as an image file."""
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    img = unnormalize(tensor).clamp(0, 1)
    img = transforms.ToPILImage()(img.cpu())
    img.save(path)


def fgsm_attack(image, epsilon, data_grad):
    """Create adversarial example using FGSM"""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def generate_adversarial_example(model, device, dataloader, epsilon):
    for i, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        final_pred = model(perturbed_data).max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            continue

        class_folder = os.path.join(output_dir, str(target.item()))
        os.makedirs(class_folder, exist_ok=True)
        image_path = os.path.join(class_folder, f"{i}.png")
        save_image(perturbed_data.squeeze(), image_path)
        logging.info(f"Saved {image_path} with epsilon={epsilon}")


# 设定扰动大小（epsilon）
epsilon = 0.01  # 这是一个非常小的值，可以根据需要调整

# 生成对抗样本
generate_adversarial_example(model, device, val_loader, epsilon)

print('Adversarial samples generation complete.')