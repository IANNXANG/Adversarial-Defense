import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
import os
import torchvision.utils as vutils

# 配置日志记录
logging.basicConfig(filename='accuracy_attack.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 加载预训练的ResNet50模型
model = models.resnet50(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型移动到设备上
model.eval()  # 设置为评估模式

# 定义数据转换（和之前保持一致，用于处理原始图像数据）
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载ImageNet验证集（原始的，未添加对抗样本的）
val_dataset = datasets.ImageFolder(root='ILSVRC2012_img_val_categories', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# FGSM攻击函数，用于生成对抗样本
def fgsm_attack(image, target, model, epsilon=0.03):
    """
    使用FGSM生成对抗样本
    :param image: 输入图像张量，已经归一化并且在相应设备上
    :param target: 对应的真实标签张量，在相应设备上
    :param model: 要攻击的模型
    :param epsilon: 扰动的强度系数
    :return: 生成的对抗样本张量
    """
    image.requires_grad = True
    output = model(image)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()
    # 获取梯度的符号
    sign_data_grad = image.grad.data.sign()
    # 生成对抗样本
    perturbed_image = image + epsilon * sign_data_grad
    # 进行裁剪，确保数据在合理范围（例如0-1之间，根据具体归一化和数据范围调整）
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image.detach()

# 定义保存对抗样本数据集的文件夹路径，这里假设在当前目录下创建名为'adversarial_samples'的文件夹
save_folder_path = "adversarial_samples"
os.makedirs(save_folder_path, exist_ok=True)

# 用于保存对抗样本数据集的转换（这里只包含ToTensor和Normalize，去掉了Resize和CenterCrop，因为我们希望保存处理后的固定尺寸样本）
save_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 遍历原始验证集数据，生成对抗样本并保存
for batch_idx, (data, target) in enumerate(val_loader):
    data, target = data.to(device), target.to(device)
    # 生成对抗样本
    adversarial_data = fgsm_attack(data, target, model)
    for i in range(adversarial_data.size(0)):
        # 将对抗样本转换回PIL图像格式（方便后续保存等操作）
        img = transforms.ToPILImage()(adversarial_data[i].cpu())
        # 获取原始图像对应的类别文件夹名称（假设原始数据集是按类别分文件夹存放的）
        class_folder = val_dataset.classes[target[i].item()]
        class_folder_path = os.path.join(save_folder_path, class_folder)
        os.makedirs(class_folder_path, exist_ok=True)
        # 保存图像到对应的类别文件夹下，文件名可以按照一定规则命名，这里简单用批次索引和样本索引组合
        save_path = os.path.join(class_folder_path, f"{batch_idx}_{i}.jpg")
        img.save(save_path)
        # 可以选择同时保存处理后的张量格式（比如如果后续可能直接用张量形式读取数据）
        tensor_save_path = os.path.join(class_folder_path, f"{batch_idx}_{i}.pt")
        torch.save(save_transform(adversarial_data[i].cpu()), tensor_save_path)

print("对抗样本数据集生成并保存完成。")