import cv2
import numpy as np
import imageio
from PIL import Image

# 读取两张图像
image1 = cv2.imread('test08.png')
image2 = cv2.imread('test08_out.png')

# 将图像转换为相同的尺寸
width, height = image2.shape[1], image2.shape[0]
image1 = cv2.resize(image1, (width, height))

# 设置滑窗大小和步长
window_size = 600
step = 5

# 创建一个新的图像对象列表
images = []

# 生成带滑窗效果的图像序列
for x in range(0, width - window_size + 1, step):
    # 图像复制
    blended_image = np.copy(image1)

    # 滑窗区域替换
    window_region = image2[:, x:x+window_size]
    blended_image[:, x:x+window_size] = window_region

    images.append(Image.fromarray(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)))

# 保存为GIF动画
imageio.mimsave('sliding_window.gif', images, duration=0.1)
