import numpy as np
import matplotlib.pyplot as plt

# 创建一个简单的黑白格子图像（8x8像素）
image = np.zeros((8, 8))  # 创建一个8x8的全0数组
image[::2, 1::2] = 1  # 将奇数行、偶数列的元素设为1
image[1::2, ::2] = 1  # 将偶数行、奇数列的元素设为1

# 使用Matplotlib库展示图像
plt.imshow(image, cmap='gray', interpolation='nearest')
plt.axis('off')  # 不显示坐标轴
plt.title('Sample Image')
plt.show()
