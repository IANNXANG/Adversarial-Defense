
import os

# 打开文件
with open('C:/Users/10655/Desktop/code/人工智能安全/Adversarial-Attack/mk_datasets/mkdir.txt', 'r') as f:
    # 读取每一行
    for line in f:
        # 去掉行尾的换行符
        line = line.strip()
        # 如果是空行，跳过
        if not line:
            continue
        # 执行命令
        os.system(line)

import sys
# import cv2
import matplotlib.pyplot as plt
import re, os
from PIL import Image
import numpy as np

lines = []

# 这个是放categories.txt的路径
for each in open(r"categories.txt", "r"):
    lines.append((each[3:31], each[32:41]))

# print(len(lines))

for i, item in enumerate(lines):
    if i % 100 == 0:
        print(i, 'done')

    # 这个是获取原始验证集ILSVRC2012_img_val中每个图像的路径
    image = Image.open('ImageNet/ILSVRC2012_img_val/{}'.format(item[0]))

    # 这个是上面的代码里放一千个空文件夹的路径
    path = 'categories/{}'.format(item[1])
    image.save(os.path.join(path, item[0]))
    image.close()
