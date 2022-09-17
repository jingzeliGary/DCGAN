'''
加载训练好的 G_model, 将随机点生成人脸图像
'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from model.Generator_model import G_model


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 加载模型
G_model = G_model()
G_model.to(device)
# 加载训练后的权重
G_model.load_state_dict(torch.load('./save_model/g_model.pth'))
G_model.eval()

# 生成随机点
fixed_noise = torch.randn(16, 100, 1, 1, device=device)
fake_imgs = G_model(fixed_noise).detach().cpu().numpy()

fig = plt.figure(figsize=(10, 10))
for i in range(fake_imgs.shape[0]):
    plt.subplot(4, 4, i+1)
    img =  np.transpose(fake_imgs[i],(1,2,0))
    img =(img+1 )/ 2 * 255
    img = img.astype('int')
    plt.imshow(img)
    plt.axis('off')
plt.show()
