import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging

def main_resnet50():
    # 配置日志记录
    logging.basicConfig(filename='accuracy_resnet50.log', level=logging.INFO,
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

    # 定义函数来计算准确率
    def calculate_accuracy(model, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)  # 将数据移动到设备上
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                print(f"batch {i + 1}, 准确率: {correct / total * 100:.2f}%")
                logging.info(f"batch {i + 1}, 准确率: {correct / total * 100:.2f}%")
        return correct / total

    # 计算准确率
    accuracy = calculate_accuracy(model, val_loader)
    print(f"The accuracy of ResNet50 on the ImageNet validation set is: {accuracy * 100:.2f}%")
    logging.info(f"The accuracy of ResNet50 on the ImageNet validation set is: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main_resnet50()