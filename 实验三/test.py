# 实验三：基于神经网络的手写数字识别实验
# 全连接神经网络与卷积神经网络（CNN）模型比较
# 作者：苏凯军
# 说明：本代码包含任务零（数据集分析）、任务一（全连接神经网络）、
#       任务二（CNN）、任务三（模型比较）的完整实现

# ==========================================
# 步骤 1：导入必要的库
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import os

# 设置中文字体（解决中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子，保证实验结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有 GPU 可用，优先使用 GPU 加速训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")
if torch.cuda.is_available():
    print(f"GPU 型号：{torch.cuda.get_device_name(0)}")

# 创建保存图片的目录
save_dir = "./data"
os.makedirs(save_dir, exist_ok=True)

# ==========================================
# 任务零：实验任务理解与数据集分析（15 分）
# ==========================================
print("\n" + "=" * 50)
print("任务零：实验任务理解与数据集分析")
print("=" * 50)

# 1) 使用 torchvision 加载 MNIST 数据集
# 数据预处理：将图像转换为张量，并归一化到 [0, 1] 范围
transform = transforms.Compose([
    transforms.ToTensor(),  # 将 PIL Image 转换为 Tensor，并自动归一化到 [0,1]
])

# 下载并加载训练集和测试集
# 如果本地没有数据，会自动从网络下载到 ./data/MNIST 目录
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

print(f"\n【数据集基本信息】")
print(f"训练集样本数量：{len(train_dataset)}")
print(f"测试集样本数量：{len(test_dataset)}")
print(f"图像类别数：10（数字 0~9）")
print(f"单张图像尺寸：28 × 28 像素，灰度图")

# 2) 查看单张图片的形状和数据表示方式
sample_image, sample_label = train_dataset[0]
print(f"\n【单张图片信息】")
print(f"单张图片张量形状（shape）：{sample_image.shape}")
print(f"对应标签：{sample_label}")
print(f"像素值范围：{sample_image.min():.4f} ~ {sample_image.max():.4f}")
print(f"解释：1×28×28 表示 1 个通道（灰度），28×28 像素")

# 3) 随机显示若干张手写数字图片
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('MNIST 手写数字样本展示', fontsize=16)
for i, ax in enumerate(axes.flat):
    # 随机选取一张图片
    idx = np.random.randint(0, len(train_dataset))
    img, label = train_dataset[idx]
    # 张量形状为 (1, 28, 28)，需要去掉通道维度并转为 numpy
    ax.imshow(img.squeeze().numpy(), cmap='gray')
    ax.set_title(f'标签: {label}')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{save_dir}/任务零_MNIST样本展示.png', dpi=150, bbox_inches='tight')
print(f"\n已保存样本展示图：{save_dir}/任务零_MNIST样本展示.png")
plt.close()

# 4) 统计训练集中各数字类别的数量，分析数据集均衡性
labels = [label for _, label in train_dataset]
unique, counts = np.unique(labels, return_counts=True)
print(f"\n【训练集各类别数量统计】")
for u, c in zip(unique, counts):
    print(f"数字 {u}: {c} 张")
print(f"数据集均衡性分析：各类别数量接近，数据集较为均衡")

# 可视化类别分布
plt.figure(figsize=(10, 5))
plt.bar(unique, counts, color='steelblue', edgecolor='black')
plt.xlabel('数字类别')
plt.ylabel('样本数量')
plt.title('MNIST 训练集各类别数量分布')
plt.xticks(unique)
for i, v in enumerate(counts):
    plt.text(i, v + 50, str(v), ha='center', va='bottom')
plt.savefig(f'{save_dir}/任务零_类别分布.png', dpi=150, bbox_inches='tight')
print(f"已保存类别分布图：{save_dir}/任务零_类别分布.png")
plt.close()

# 5) 从训练集中划分验证集（用于监控训练过程中的模型表现）
# 将 60000 张训练图片划分为 50000 张训练 + 10000 张验证
train_size = 50000
val_size = 10000
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# 6) 设置批量大小 batch_size，使用 DataLoader 批量加载数据
batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\n【数据加载器信息】")
print(f"训练批次数量：{len(train_loader)}（每批 {batch_size} 张）")
print(f"验证批次数量：{len(val_loader)}")
print(f"测试批次数量：{len(test_loader)}")

print("\n任务零完成！MNIST 数据集加载与探索结束。\n")

# ==========================================
# 任务一：全连接神经网络建模（25 分）
# ==========================================
print("=" * 50)
print("任务一：全连接神经网络建模——手写数字识别")
print("=" * 50)

# 1) 定义全连接神经网络模型
class FullyConnectedNN(nn.Module):
    """
    全连接神经网络（FCNN）模型
    输入：展平后的 784 维像素向量（28×28=784）
    输出：10 个类别的预测得分（对应数字 0~9）
    """
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        # 展平层：将 1×28×28 的图像张量展平为 784 维向量
        self.flatten = nn.Flatten()
        # 隐藏层 1：784 → 128，使用 ReLU 激活函数
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        # 隐藏层 2：128 → 64，使用 ReLU 激活函数
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        # 输出层：64 → 10（10 个数字类别）
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)      # 展平操作
        x = self.relu1(self.fc1(x))  # 第一层 + ReLU
        x = self.relu2(self.fc2(x))  # 第二层 + ReLU
        x = self.fc3(x)          # 输出层（无需激活，CrossEntropyLoss 内部包含 Softmax）
        return x

# 2) 创建模型实例并移动到设备（GPU 或 CPU）
fc_model = FullyConnectedNN().to(device)
print(f"\n【全连接神经网络结构】")
print(fc_model)

# 统计模型参数数量
total_params = sum(p.numel() for p in fc_model.parameters())
print(f"\n模型总参数量：{total_params:,}")

# 3) 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失，适用于多分类任务
optimizer_fc = optim.Adam(fc_model.parameters(), lr=0.001)  # Adam 优化器，学习率 0.001

# 4) 训练模型（5 个 epoch）
num_epochs = 5
fc_train_losses = []   # 记录每轮训练的平均损失
fc_val_losses = []     # 记录每轮验证的平均损失
fc_val_accuracies = [] # 记录每轮验证的准确率

print(f"\n开始训练全连接神经网络（{num_epochs} 个 epoch）...")
start_time = time.time()

for epoch in range(num_epochs):
    # ---------- 训练阶段 ----------
    fc_model.train()  # 设置为训练模式
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播：计算预测结果
        outputs = fc_model(images)
        loss = criterion(outputs, labels)

        # 反向传播：计算梯度并更新参数
        optimizer_fc.zero_grad()  # 清空上一步的梯度
        loss.backward()             # 计算当前损失的梯度
        optimizer_fc.step()         # 更新模型参数

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    fc_train_losses.append(avg_train_loss)

    # ---------- 验证阶段 ----------
    fc_model.eval()  # 设置为评估模式
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 验证时不计算梯度，节省内存和计算
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = fc_model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 计算预测正确的数量
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    fc_val_losses.append(avg_val_loss)
    fc_val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"训练损失: {avg_train_loss:.4f} | "
          f"验证损失: {avg_val_loss:.4f} | "
          f"验证准确率: {val_accuracy:.2f}%")

fc_train_time = time.time() - start_time
print(f"\n全连接神经网络训练完成！耗时：{fc_train_time:.2f} 秒")

# 5) 在测试集上评估全连接神经网络
fc_model.eval()
fc_all_preds = []
fc_all_labels = []
fc_test_correct = 0
fc_test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = fc_model(images)
        _, predicted = torch.max(outputs, 1)
        fc_test_total += labels.size(0)
        fc_test_correct += (predicted == labels).sum().item()
        fc_all_preds.extend(predicted.cpu().numpy())
        fc_all_labels.extend(labels.cpu().numpy())

fc_test_accuracy = 100 * fc_test_correct / fc_test_total
print(f"\n【全连接神经网络测试结果】")
print(f"测试集准确率：{fc_test_accuracy:.2f}%")
print(f"测试集样本数：{fc_test_total}")
print(f"预测正确数：{fc_test_correct}")
print(f"预测错误数：{fc_test_total - fc_test_correct}")

# 6) 绘制全连接神经网络的训练损失曲线和验证准确率曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
ax1.plot(range(1, num_epochs+1), fc_train_losses, 'b-o', label='训练损失')
ax1.plot(range(1, num_epochs+1), fc_val_losses, 'r-s', label='验证损失')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('损失值 (Loss)')
ax1.set_title('全连接神经网络——训练与验证损失曲线')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# 准确率曲线
ax2.plot(range(1, num_epochs+1), fc_val_accuracies, 'g-^', label='验证准确率')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('准确率 (%)')
ax2.set_title('全连接神经网络——验证准确率曲线')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(f'{save_dir}/任务一_FCNN损失与准确率曲线.png', dpi=150, bbox_inches='tight')
print(f"\n已保存 FCNN 损失与准确率曲线：{save_dir}/任务一_FCNN损失与准确率曲线.png")
plt.close()

# 7) 绘制混淆矩阵
cm_fc = confusion_matrix(fc_all_labels, fc_all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_fc, annot=True, fmt='d', cmap='Blues', square=True,
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title(f'全连接神经网络混淆矩阵\n(测试准确率: {fc_test_accuracy:.2f}%)')
plt.savefig(f'{save_dir}/任务一_FCNN混淆矩阵.png', dpi=150, bbox_inches='tight')
print(f"已保存 FCNN 混淆矩阵：{save_dir}/任务一_FCNN混淆矩阵.png")
plt.close()

# 8) 输出分类报告
print(f"\n【全连接神经网络分类报告】")
print(classification_report(fc_all_labels, fc_all_preds, target_names=[str(i) for i in range(10)]))

# 9) 随机显示 10 张测试图片，标注真实标签和预测标签
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('FCNN 预测结果展示（绿色=正确，红色=错误）', fontsize=16)

test_images = []
test_labels_list = []
for img, lbl in test_dataset:
    test_images.append(img)
    test_labels_list.append(lbl)

# 随机选择 10 张
indices = np.random.choice(len(test_dataset), 10, replace=False)
for idx, ax in zip(indices, axes.flat):
    img = test_images[idx].unsqueeze(0).to(device)
    true_label = test_labels_list[idx]
    with torch.no_grad():
        output = fc_model(img)
        pred_label = torch.argmax(output, 1).item()

    ax.imshow(test_images[idx].squeeze().numpy(), cmap='gray')
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f'真实: {true_label} | 预测: {pred_label}', color=color)
    ax.axis('off')

plt.tight_layout()
plt.savefig(f'{save_dir}/任务一_FCNN预测结果.png', dpi=150, bbox_inches='tight')
print(f"已保存 FCNN 预测结果图：{save_dir}/任务一_FCNN预测结果.png")
plt.close()

print("\n任务一完成！全连接神经网络建模结束。\n")

# ==========================================
# 任务二：卷积神经网络建模（30 分）
# ==========================================
print("=" * 50)
print("任务二：卷积神经网络（CNN）建模——手写数字识别")
print("=" * 50)

# 1) 定义卷积神经网络模型
class CNN(nn.Module):
    """
    卷积神经网络（CNN）模型
    输入：1×28×28 的灰度图像张量
    通过卷积层和池化层提取局部空间特征，再通过全连接层分类
    """
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层 1：输入 1 通道，输出 16 通道，卷积核大小 3×3，填充 1 保持尺寸
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # 池化层 1：最大池化，池化核 2×2，特征图尺寸减半（28→14）
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层 2：输入 16 通道，输出 32 通道，卷积核大小 3×3，填充 1 保持尺寸
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # 池化层 2：最大池化，特征图尺寸再次减半（14→7）
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 展平层：将多维特征图展平为一维向量
        self.flatten = nn.Flatten()

        # 全连接层 1：32×7×7 = 1568 维 → 128 维
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        # 输出层：128 → 10（10 个数字类别）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))  # 卷积1 → ReLU → 池化1
        x = self.pool2(self.relu2(self.conv2(x)))  # 卷积2 → ReLU → 池化2
        x = self.flatten(x)                        # 展平
        x = self.relu3(self.fc1(x))                # 全连接1 + ReLU
        x = self.fc2(x)                            # 输出层
        return x

# 2) 创建 CNN 模型实例
cnn_model = CNN().to(device)
print(f"\n【卷积神经网络结构】")
print(cnn_model)

# 统计 CNN 模型参数数量
cnn_total_params = sum(p.numel() for p in cnn_model.parameters())
print(f"\nCNN 模型总参数量：{cnn_total_params:,}")
print(f"（FCNN 参数量：{total_params:,}，CNN 参数量：{cnn_total_params:,}）")

# 3) 设置损失函数和优化器
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001)

# 4) 训练 CNN 模型（5 个 epoch）
cnn_train_losses = []
cnn_val_losses = []
cnn_val_accuracies = []

print(f"\n开始训练 CNN（{num_epochs} 个 epoch）...")
start_time_cnn = time.time()

for epoch in range(num_epochs):
    # ---------- 训练阶段 ----------
    cnn_model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = cnn_model(images)
        loss = criterion(outputs, labels)

        optimizer_cnn.zero_grad()
        loss.backward()
        optimizer_cnn.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    cnn_train_losses.append(avg_train_loss)

    # ---------- 验证阶段 ----------
    cnn_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = cnn_model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    cnn_val_losses.append(avg_val_loss)
    cnn_val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"训练损失: {avg_train_loss:.4f} | "
          f"验证损失: {avg_val_loss:.4f} | "
          f"验证准确率: {val_accuracy:.2f}%")

cnn_train_time = time.time() - start_time_cnn
print(f"\nCNN 训练完成！耗时：{cnn_train_time:.2f} 秒")

# 5) 在测试集上评估 CNN
cnn_model.eval()
cnn_all_preds = []
cnn_all_labels = []
cnn_test_correct = 0
cnn_test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs, 1)
        cnn_test_total += labels.size(0)
        cnn_test_correct += (predicted == labels).sum().item()
        cnn_all_preds.extend(predicted.cpu().numpy())
        cnn_all_labels.extend(labels.cpu().numpy())

cnn_test_accuracy = 100 * cnn_test_correct / cnn_test_total
print(f"\n【CNN 测试结果】")
print(f"测试集准确率：{cnn_test_accuracy:.2f}%")
print(f"测试集样本数：{cnn_test_total}")
print(f"预测正确数：{cnn_test_correct}")
print(f"预测错误数：{cnn_test_total - cnn_test_correct}")

# 6) 绘制 CNN 的训练损失曲线和验证准确率曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, num_epochs+1), cnn_train_losses, 'b-o', label='训练损失')
ax1.plot(range(1, num_epochs+1), cnn_val_losses, 'r-s', label='验证损失')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('损失值 (Loss)')
ax1.set_title('CNN——训练与验证损失曲线')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(range(1, num_epochs+1), cnn_val_accuracies, 'g-^', label='验证准确率')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('准确率 (%)')
ax2.set_title('CNN——验证准确率曲线')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(f'{save_dir}/任务二_CNN损失与准确率曲线.png', dpi=150, bbox_inches='tight')
print(f"\n已保存 CNN 损失与准确率曲线：{save_dir}/任务二_CNN损失与准确率曲线.png")
plt.close()

# 7) 绘制 CNN 混淆矩阵
cm_cnn = confusion_matrix(cnn_all_labels, cnn_all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Greens', square=True,
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title(f'CNN 混淆矩阵\n(测试准确率: {cnn_test_accuracy:.2f}%)')
plt.savefig(f'{save_dir}/任务二_CNN混淆矩阵.png', dpi=150, bbox_inches='tight')
print(f"已保存 CNN 混淆矩阵：{save_dir}/任务二_CNN混淆矩阵.png")
plt.close()

# 8) 输出 CNN 分类报告
print(f"\n【CNN 分类报告】")
print(classification_report(cnn_all_labels, cnn_all_preds, target_names=[str(i) for i in range(10)]))

# 9) 显示若干预测正确和预测错误的样本
correct_indices = []
error_indices = []

for i in range(len(test_dataset)):
    img = test_images[i].unsqueeze(0).to(device)
    true_label = test_labels_list[i]
    with torch.no_grad():
        output = cnn_model(img)
        pred_label = torch.argmax(output, 1).item()
    if pred_label == true_label:
        correct_indices.append((i, true_label, pred_label))
    else:
        error_indices.append((i, true_label, pred_label))
    # 各收集足够的样本即可停止遍历
    if len(correct_indices) >= 10 and len(error_indices) >= 10:
        break

# 展示预测正确的样本
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('CNN 预测正确的样本', fontsize=16)
for idx, ax in zip(range(min(10, len(correct_indices))), axes.flat):
    i, true_l, pred_l = correct_indices[idx]
    ax.imshow(test_images[i].squeeze().numpy(), cmap='gray')
    ax.set_title(f'真实: {true_l} | 预测: {pred_l}', color='green')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{save_dir}/任务二_CNN正确样本.png', dpi=150, bbox_inches='tight')
print(f"已保存 CNN 正确样本图：{save_dir}/任务二_CNN正确样本.png")
plt.close()

# 展示预测错误的样本
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('CNN 预测错误的样本', fontsize=16)
for idx, ax in zip(range(min(10, len(error_indices))), axes.flat):
    i, true_l, pred_l = error_indices[idx]
    ax.imshow(test_images[i].squeeze().numpy(), cmap='gray')
    ax.set_title(f'真实: {true_l} | 预测: {pred_l}', color='red')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{save_dir}/任务二_CNN错误样本.png', dpi=150, bbox_inches='tight')
print(f"已保存 CNN 错误样本图：{save_dir}/任务二_CNN错误样本.png")
plt.close()

print("\n任务二完成！CNN 建模结束。\n")

# ==========================================
# 任务三：模型比较与结果分析（20 分）
# ==========================================
print("=" * 50)
print("任务三：模型比较与结果分析")
print("=" * 50)

# 1) 打印模型比较信息
print(f"\n【模型训练信息对比】")
print(f"{'指标':<25} {'全连接神经网络':<20} {'CNN':<20}")
print(f"{'-'*65}")
print(f"{'训练轮数 (epoch)':<25} {num_epochs:<20} {num_epochs:<20}")
print(f"{'训练时间 (秒)':<25} {fc_train_time:.2f}{' '*15} {cnn_train_time:.2f}")
print(f"{'模型参数量':<25} {total_params:<20,} {cnn_total_params:<20,}")
print(f"{'测试集准确率 (%)':<25} {fc_test_accuracy:.2f}{' '*15} {cnn_test_accuracy:.2f}")
print(f"{'测试集正确数':<25} {fc_test_correct:<20} {cnn_test_correct:<20}")
print(f"{'测试集错误数':<25} {fc_test_total-fc_test_correct:<20} {cnn_test_total-cnn_test_correct:<20}")

# 2) 将两种模型的训练损失曲线绘制在同一张图中
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), fc_train_losses, 'b-o', label='FCNN 训练损失')
plt.plot(range(1, num_epochs+1), fc_val_losses, 'b--s', label='FCNN 验证损失')
plt.plot(range(1, num_epochs+1), cnn_train_losses, 'r-^', label='CNN 训练损失')
plt.plot(range(1, num_epochs+1), cnn_val_losses, 'r--v', label='CNN 验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失值 (Loss)')
plt.title('FCNN 与 CNN 训练/验证损失曲线对比')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'{save_dir}/任务三_损失曲线对比.png', dpi=150, bbox_inches='tight')
print(f"\n已保存损失曲线对比图：{save_dir}/任务三_损失曲线对比.png")
plt.close()

# 3) 将两种模型的验证准确率曲线绘制在同一张图中
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), fc_val_accuracies, 'b-o', label='FCNN 验证准确率')
plt.plot(range(1, num_epochs+1), cnn_val_accuracies, 'r-^', label='CNN 验证准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率 (%)')
plt.title('FCNN 与 CNN 验证准确率曲线对比')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'{save_dir}/任务三_准确率曲线对比.png', dpi=150, bbox_inches='tight')
print(f"已保存准确率曲线对比图：{save_dir}/任务三_准确率曲线对比.png")
plt.close()

# 4) 并列绘制两个模型的混淆矩阵
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

sns.heatmap(cm_fc, annot=True, fmt='d', cmap='Blues', square=True,
            xticklabels=range(10), yticklabels=range(10), ax=ax1)
ax1.set_title(f'全连接神经网络混淆矩阵\n准确率: {fc_test_accuracy:.2f}%')
ax1.set_xlabel('预测标签')
ax1.set_ylabel('真实标签')

sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Greens', square=True,
            xticklabels=range(10), yticklabels=range(10), ax=ax2)
ax2.set_title(f'CNN 混淆矩阵\n准确率: {cnn_test_accuracy:.2f}%')
ax2.set_xlabel('预测标签')
ax2.set_ylabel('真实标签')

plt.tight_layout()
plt.savefig(f'{save_dir}/任务三_混淆矩阵对比.png', dpi=150, bbox_inches='tight')
print(f"已保存混淆矩阵对比图：{save_dir}/任务三_混淆矩阵对比.png")
plt.close()

# 5) 分析模型容易识别错误的数字类别
# 计算每个数字的错误率
fc_errors_per_class = []
cnn_errors_per_class = []
for i in range(10):
    # FCNN 错误数 = 总数 - 正确数（对角线元素）
    fc_total_i = cm_fc[i, :].sum()
    fc_correct_i = cm_fc[i, i]
    fc_errors_per_class.append(fc_total_i - fc_correct_i)

    cnn_total_i = cm_cnn[i, :].sum()
    cnn_correct_i = cm_cnn[i, i]
    cnn_errors_per_class.append(cnn_total_i - cnn_correct_i)

print(f"\n【各数字类别的预测错误数】")
print(f"{'数字':<10} {'FCNN 错误数':<15} {'CNN 错误数':<15}")
print(f"{'-'*40}")
for i in range(10):
    print(f"{i:<10} {fc_errors_per_class[i]:<15} {cnn_errors_per_class[i]:<15}")

# 找出最容易混淆的数字对（混淆矩阵非对角线最大值）
fc_confusion_pairs = []
cnn_confusion_pairs = []
for i in range(10):
    for j in range(10):
        if i != j:
            fc_confusion_pairs.append((i, j, cm_fc[i, j]))
            cnn_confusion_pairs.append((i, j, cm_cnn[i, j]))

fc_confusion_pairs.sort(key=lambda x: x[2], reverse=True)
cnn_confusion_pairs.sort(key=lambda x: x[2], reverse=True)

print(f"\n【FCNN 最容易混淆的数字对（Top 5）】")
for i, j, count in fc_confusion_pairs[:5]:
    print(f"真实 {i} → 预测 {j}：{count} 次")

print(f"\n【CNN 最容易混淆的数字对（Top 5）】")
for i, j, count in cnn_confusion_pairs[:5]:
    print(f"真实 {i} → 预测 {j}：{count} 次")

# 6) 显示预测错误样本（FCNN）
fc_error_indices = []
for i in range(len(test_dataset)):
    img = test_images[i].unsqueeze(0).to(device)
    true_label = test_labels_list[i]
    with torch.no_grad():
        output = fc_model(img)
        pred_label = torch.argmax(output, 1).item()
    if pred_label != true_label:
        fc_error_indices.append((i, true_label, pred_label))
    if len(fc_error_indices) >= 10:
        break

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('FCNN 预测错误的样本（真实标签 → 预测标签）', fontsize=16)
for idx, ax in zip(range(min(10, len(fc_error_indices))), axes.flat):
    i, true_l, pred_l = fc_error_indices[idx]
    ax.imshow(test_images[i].squeeze().numpy(), cmap='gray')
    ax.set_title(f'{true_l} → {pred_l}', color='red')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{save_dir}/任务三_FCNN错误样本分析.png', dpi=150, bbox_inches='tight')
print(f"\n已保存 FCNN 错误样本分析图：{save_dir}/任务三_FCNN错误样本分析.png")
plt.close()

# 7) 从输入表示、特征提取、参数数量、训练效果和可解释性角度比较两种模型
print(f"\n【模型综合比较】")
print(f"{'比较维度':<20} {'全连接神经网络 (FCNN)':<40} {'卷积神经网络 (CNN)':<40}")
print(f"{'-'*100}")
print(f"{'输入表示':<20} {'展平为 784 维向量，丢失空间信息':<40} {'保留 1×28×28 图像张量结构':<40}")
print(f"{'特征提取':<20} {'全连接层学习全局组合特征':<40} {'卷积层提取局部空间特征':<40}")
print(f"{'参数量':<20} {f'{total_params:,}':<40} {f'{cnn_total_params:,}':<40}")
print(f"{'训练效果':<20} {f'测试准确率 {fc_test_accuracy:.2f}%':<40} {f'测试准确率 {cnn_test_accuracy:.2f}%':<40}")
print(f"{'可解释性':<20} {'特征含义不明确':<40} {'卷积核可可视化，特征可解释':<40}")

print("\n任务三完成！模型比较与结果分析结束。\n")

# ==========================================
# 实验结束，汇总输出
# ==========================================
print("=" * 50)
print("实验三全部任务完成！")
print("=" * 50)
print(f"\n【实验结果汇总】")
print(f"全连接神经网络测试准确率：{fc_test_accuracy:.2f}%")
print(f"卷积神经网络测试准确率：{cnn_test_accuracy:.2f}%")
print(f"准确率提升：{cnn_test_accuracy - fc_test_accuracy:.2f}%")
print(f"\n所有结果图片已保存到：{save_dir}/")
print("\n实验结束，感谢使用！")

