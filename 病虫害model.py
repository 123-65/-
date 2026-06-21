"""
基于卷积神经网络的农业病虫害识别（复现论文方法）
数据集：NGLD (葡萄叶片病害四分类)
参考文献：李子涵，等. 基于卷积神经网络的农业病虫害识别研究综述. 江苏农业科学，2023
"""

import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------- 固定随机种子 --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# -------------------- 参数设置 --------------------
DATA_ROOT = "./NGLD"          # 数据集根目录，子文件夹为类别名
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001        # 参考 Thenmozhi 等发现 0.0001 效果较好 [6]
NUM_CLASSES = 4
IMG_SIZE = 224                # 标准输入尺寸
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED = True             # 是否使用预训练权重

# -------------------- 自定义数据集 --------------------
class NGLDDataset(Dataset):
    """
    NGLD数据集加载器，使用os.walk递归扫描所有子目录，
    自动将包含图片的文件夹名称作为类别标签。
    支持任意层级结构，只要叶子目录名就是类别名。
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.classes = []

        # 使用字典存储每个类别的图片路径列表
        class_to_images = {}
        for dirpath, _, filenames in os.walk(root):
            # 筛选图片文件
            img_files = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if img_files:
                # 取当前目录名作为类别名
                class_name = os.path.basename(dirpath)
                if class_name not in class_to_images:
                    class_to_images[class_name] = []
                for fname in img_files:
                    class_to_images[class_name].append(os.path.join(dirpath, fname))

        if not class_to_images:
            raise ValueError(f"No images found in {root}. Please check directory structure.")

        # 构建类别映射
        self.classes = sorted(class_to_images.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for cls, img_list in class_to_images.items():
            for img_path in img_list:
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[cls])

        print(f"Loaded {len(self.images)} images, classes: {self.classes}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------- 数据预处理（论文表2） --------------------
# 包括：调整图像尺寸、归一化、灰度化、PCA白化、颜色模型转换、降噪、图像分割
# 这里实现常用操作，灰度化和PCA白化作为可选，在增强中体现

# 基础变换（必选）
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),          # 调整图像尺寸 [9-20,27,29-30,10]
    transforms.ToTensor(),                            # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 归一化，ImageNet标准 [4-15,25]
                         std=[0.229, 0.224, 0.225])
])

# -------------------- 数据增强（论文表4及具体文献） --------------------
# 实现有监督增强：几何变换、空间变换、颜色变换
# 无监督增强（GAN/AutoAugment）这里不实现，但可留作扩展

class CustomAugmentation:
    """封装多种增强方法，并引用论文中的具体文献"""
    @staticmethod
    def duan_augmentation(image):
        """
        参考文献：Duan 等 [31]
        在倍率为0.2的范围内对图像进行剪切和缩放，生成多个增强图像
        这里模拟：随机缩放0.8~1.2，然后中心裁剪回原尺寸
        """
        scale = 1.0 + random.uniform(-0.2, 0.2)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        resized = F.resize(image, (new_h, new_w))
        # 随机裁剪回原尺寸
        i, j, h, w = transforms.RandomCrop.get_params(resized, output_size=(h, w))
        cropped = F.crop(resized, i, j, h, w)
        return cropped

    @staticmethod
    def sladojevic_augmentation(image):
        """
        参考文献：Sladojevic 等 [23]
        对单幅图像进行仿射变换与透视变换，增加轻微失真，减少过拟合
        """
        # 仿射变换参数（均为范围，由 RandomAffine 内部随机采样）
        angle = (-10, 10)  # 旋转角度范围（度）
        translate = (0.1, 0.1)  # 水平/垂直最大平移比例（相对于图像尺寸）
        scale = (0.9, 1.1)  # 缩放系数范围
        shear = (-5, 5)  # 剪切角度范围（度）
        affine = transforms.RandomAffine(degrees=angle, translate=translate,
                                         scale=scale, shear=shear)
        image = affine(image)
        # 透视变换（轻微）
        perspective = transforms.RandomPerspective(distortion_scale=0.1, p=1.0)
        image = perspective(image)

        return image

    @staticmethod
    def liu_augmentation(image):
        """
        参考文献：Liu 等 [17]
        随机增加或减少像素的RGB值调整亮度，根据亮度中值调整对比度
        这里使用torchvision的ColorJitter实现亮度和对比度调整
        """
        # 亮度调整：随机±0.2，对比度调整：随机±0.2
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
        return color_jitter(image)

# 训练集增强组合（包含论文中的多种方法）
train_transforms = transforms.Compose([
    # 基础几何变换（裁剪、翻转、旋转、缩放） [1,13,15,17-18,22,26-27,30-31,34-35,37,41,45]
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),   # 随机裁剪缩放
    transforms.RandomHorizontalFlip(p=0.5),                     # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.3),                       # 随机垂直翻转
    transforms.RandomRotation(degrees=20),                      # 随机旋转

    # 空间变换（仿射、透视） [3] - 结合Sladojevic方法
    # 由于RandomAffine和RandomPerspective在组合中可能冲突，用Lambda调用自定义函数
    transforms.Lambda(lambda img: CustomAugmentation.sladojevic_augmentation(img)),

    # 颜色变换（噪声、模糊、颜色变换、擦除、填充） [3,17,20,26,24]
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # 添加随机噪声（通过自定义变换）
    transforms.Lambda(lambda img: CustomAugmentation.liu_augmentation(img)),  # Liu的亮度对比度调整
    transforms.ToTensor(),
    # 随机擦除（填充） [24]
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
    # 归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集仅使用基础预处理（无增强）
val_transforms = base_transform

# -------------------- 自定义DCNN模型（Cubuk等改进） --------------------
"""
参考文献：Cubuk 等 [14] 提出 DCNN 模型
改进：改变ReLU层的顺序，将ReLU放在2个卷积层之间，每个block末尾设置池化层
这里以ResNet为基础进行改进，但论文中未指定，我们可以实现一个简单的DCNN块
"""
class DCNNBlock(nn.Module):
    """一个改进的卷积块：Conv -> ReLU -> Conv -> ReLU -> Pool"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DCNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        return x

class DCNN(nn.Module):
    """简单DCNN实现，包含3个DCNNBlock，用于小数据集"""
    def __init__(self, num_classes=4):
        super(DCNN, self).__init__()
        self.block1 = DCNNBlock(3, 32)
        self.block2 = DCNNBlock(32, 64)
        self.block3 = DCNNBlock(64, 128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -------------------- 迁移学习模型（预训练 + 微调） --------------------
def get_model(model_name='resnet50', num_classes=4, pretrained=True):
    """
    支持多种模型：AlexNet, VGG, ResNet, MobileNet, Inception, DenseNet
    根据论文表5选择，这里默认ResNet50，用于演示
    """
    if model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        # 替换顶层全连接层 [2-14,18,25-26,45]
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'inception_v3':
        model = models.inception_v3(weights='IMAGENET1K_V1' if pretrained else None, aux_logits=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Unsupported model")
    return model

# 也可以使用自定义DCNN（不预训练）
# model = DCNN(num_classes=NUM_CLASSES)

# -------------------- 训练函数 --------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({'loss': loss.item()})
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# -------------------- 主程序 --------------------
def main():
    print(f"当前使用的设备: {DEVICE}")
    # 加载数据集
    full_dataset = NGLDDataset(DATA_ROOT, transform=None)  # 先不应用变换，后续拆分再应用
    # 划分训练/验证集 (80%/20%)
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    split = int(0.8 * num_samples)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # 注意：每个样本应用不同的变换，需在__getitem__中指定变换
    # 我们用Subset和分别定义transform
    class CustomSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_dataset = CustomSubset(train_subset, train_transforms)
    val_dataset = CustomSubset(val_subset, val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 模型
    model = get_model('resnet50', NUM_CLASSES, pretrained=PRETRAINED)
    model = model.to(DEVICE)

    # 定义损失函数和优化器 (Adam optimizer) [7,25]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 训练循环
    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # 学习率调整
        scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model.")

    print(f"\nBest validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()