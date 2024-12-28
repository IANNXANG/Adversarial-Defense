import torch
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import logging

def main_attack():
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
    output_dir = 'fgsm_images'
    os.makedirs(output_dir, exist_ok=True)

    def save_image(tensor, path):
        """Helper function to save a tensor as an image file with proper unnormalization."""
        # Define the unnormalize transform
        unnormalize = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.ToPILImage()  # Convert back to PIL Image
        ])

        # Unnormalize and clamp the tensor to [0, 1]
        img = unnormalize(tensor.cpu())
        if not isinstance(img, Image.Image):  # Ensure it's a PIL Image
            img = transforms.ToPILImage()(img)

        # Save the image in PNG format without compression
        img.save(path, format='PNG', quality=100)

    def fgsm_attack(image, epsilon, data_grad):
        """Create adversarial example using FGSM"""
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        # perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def generate_adversarial_example(model, device, dataloader, epsilon, dataset):
        # 使用dataset.classes和dataset.class_to_idx来获取类别信息
        classes = dataset.classes
        class_to_idx = dataset.class_to_idx

        # 创建输出根目录
        output_root = 'fgsm_images'
        os.makedirs(output_root, exist_ok=True)

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

            # 通过查找classes列表得到原始类别名称
            original_class_name = classes[target.item()]
            class_folder = os.path.join(output_root, original_class_name)
            os.makedirs(class_folder, exist_ok=True)

            # 确保文件名唯一，可以结合原路径或添加额外标识符
            original_sample_path = dataset.samples[i][0]  # 获取原始样本路径
            image_filename = os.path.basename(original_sample_path)
            image_path = os.path.join(class_folder, image_filename)

            save_image(perturbed_data.squeeze(), image_path)
            print(f"Saved {image_path} with epsilon={epsilon}")
            logging.info(f"Saved {image_path} with epsilon={epsilon}")

    # 设定扰动大小（epsilon）
    epsilon = 0.01  # 这是一个非常小的值，可以根据需要调整

    # 调用时传入val_dataset以访问类别信息
    generate_adversarial_example(model, device, val_loader, epsilon, val_dataset)

    print('Adversarial samples generation complete.')

if __name__ == '__main__':
    main_attack()