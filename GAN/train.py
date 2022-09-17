'''
训练
'''
import os.path
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from model.Generator_model import G_model
from model.Detector_model import D_model
import numpy as np
import matplotlib.pyplot as plt


img_preprocess = transforms.Compose([
    # 缩放，宽度64，高度自适应
    transforms.Resize(64),
    # 中心裁剪为 64 x 64 的正方形
    transforms.CenterCrop(64),
    # PIL图像转为tensor，归一化到[0,1]：Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transforms.ToTensor(),
    # 规范化至 [-1,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

])
# 按文件夹读取图片
dataset = datasets.ImageFolder(root='./data/',transform=img_preprocess)
dataloader = torch.utils.data.DataLoader(dataset,batch_size= 128,shuffle=True)

# 实例化 生成器和检测器
G_model = G_model()
D_model = D_model()

# 训练

epochs = 100
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Used device {}".format(device))
G_model.to(device)
D_model.to(device)
loss_fun = nn.BCELoss()  # 二值交叉熵损失
# 优化器
D_optimizer = optim.Adam(D_model.parameters(),lr=0.001)
G_optimizer = optim.Adam(G_model.parameters(),lr=0.001)


for epoch in range(epochs):
    # 获取批次图像
    start_time = time.time()
    for i, data in enumerate(dataloader):
        # ----------------------训练D：真实数据标记为1------------------------
        # 清空梯度
        D_model.zero_grad()
        # 获取数据
        imgs_batch = data[0].to(device)
        # 计算输出
        output = D_model(imgs_batch).reshape(-1)
        # batch_size，最后不可以用BATCH_SIZE，因为数据集数量可能不能被BATCH_SIZE整除
        b_size = imgs_batch.size(0)
        # 构建全1向量label
        ones_label = torch.full((b_size,), 1, dtype=torch.float, device=device)
        # 计算loss
        d_loss_real = loss_fun(output, ones_label)

        # 计算梯度
        d_loss_real.backward()
        # 反向传播优化
        D_optimizer.step()

        # -------------------训练D：假数据标记为0-------------------------------
        # 清除梯度
        D_model.zero_grad()
        # 构建随机张量
        noise_tensor = torch.randn(b_size, 100, 1, 1, device=device)
        # 生成假的图片
        generated_imgs = G_model(noise_tensor)
        # 假图片的输出，此时不需要训练G，可以detach
        output = D_model(generated_imgs.detach()).view(-1)

        # 构建全0向量
        zeros_label = torch.full((b_size,), 0, dtype=torch.float, device=device)
        # 计算loss
        d_loss_fake = loss_fun(output, zeros_label)
        # 计算梯度
        d_loss_fake.backward()
        # 优化
        D_optimizer.step()

        # ----------------------训练G 网络：假数据标记为1--------------------
        # 清除梯度
        G_model.zero_grad()
        # 随机张量
        noise_tensor = torch.randn(b_size, 100, 1, 1, device=device)
        # 生成假的图片
        generated_imgs = G_model(noise_tensor)
        # 假图片的输出，这里不可以detach，否则学习不到
        output = D_model(generated_imgs).view(-1)

        # 构建全1向量
        ones_label = torch.full((b_size,), 1, dtype=torch.float, device=device)
        # 计算loss
        g_loss = loss_fun(output, ones_label)
        # 计算梯度
        g_loss.backward()
        # 优化
        G_optimizer.step()

    # 打印训练时间
    print('第{}个epoch执行时间：{}s'.format(epoch, time.time() - start_time))

# 评估最后一个epoch输出结果
with torch.no_grad():
    # 生成16 个随机张量
    fixed_noise = torch.randn(16, 100, 1, 1, device=device)
    # 生成的图片
    fake_imgs = G_model(fixed_noise).detach().cpu().numpy()
    # 画布大小
    fig = plt.figure(figsize=(10, 10))
    for i in range(fake_imgs.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = np.transpose(fake_imgs[i], (1, 2, 0))
        img = (img + 1) / 2 * 255
        img = img.astype('int')
        plt.imshow(img)
        plt.axis('off')
    plt.show()

if not os.path.exists('./save_model'):
    os.mkdir('./save_model')

# 保存模型权重
torch.save(G_model.state_dict(), './save_model/g_model.pth')