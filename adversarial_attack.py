import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import logging

# 配置日志记录
logging.basicConfig(filename='fgsm_attack.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的ResNet50模型
model = models.resnet50(pretrained=True)
model.to(device)
model.eval()

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


# 定义FGSM攻击函数
def fgsm_attack(image, epsilon, data_grad):
    # 获取梯度符号
    sign_data_grad = data_grad.sign()
    # 创建对抗样本
    perturbed_image = image + epsilon * sign_data_grad
    # 将像素值限制在[0,1]范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# 对抗样本生成与保存
epsilon = 0.01  # 扰动强度，可以根据需求调整
save_dir = 'adversarial_examples'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for images, labels in tqdm(val_loader, desc="Creating adversarial examples"):
    images, labels = images.to(device), labels.to(device)

    images.requires_grad = True

    outputs = model(images)
    loss = cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    perturbed_images = fgsm_attack(images, epsilon, data_grad)

    for i in range(len(labels)):
        adv_image = perturbed_images[i].detach().cpu().numpy()
        adv_image = np.transpose(adv_image, (1, 2, 0))
        adv_image = adv_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        adv_image = np.clip(adv_image, 0, 1)

        original_path = val_loader.dataset.samples[val_loader.batch_sampler.sampler.indices[i]][0]
        adv_image_pil = Image.fromarray((adv_image * 255).astype(np.uint8))

        # 恢复原始分辨率
        original_image = Image.open(original_path)
        adv_image_resized = adv_image_pil.resize(original_image.size, Image.ANTIALIAS)

        # 构建保存路径
        filename = os.path.basename(original_path)
        save_path = os.path.join(save_dir, filename)

        # 以高质量保存图片
        adv_image_resized.save(save_path, quality=95)

        logging.info(f"Saved adversarial example: {filename}")

print("Adversarial example creation completed.")