## 看看模型行不行
from email.mime import image
import datetime
from keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as pyplot


model = load_model("newmodel1661091759.h5")

# 激活函数
def noise(x : float) -> float:
    i = np.floor(x)
    f = np.modf(x)[0]
    k = np.modf(i * 0.1731)[0] * 16.0 - 4.0
    f *= f - 1.0
    f *= k * f - 1.0
    f *= np.sign(np.modf(x * 0.5)[0] - 0.5)
    return f

image_width = 220
image_height = 512
image_channel = 3

# 范围重新映射为 (-1, 1)
x_test = np.zeros(shape=(image_height*image_width, 2), dtype=np.float32)
for y in range(0, image_height):
    for x in range(0, image_width):
        x_test[y * image_width + x] = np.array([2 * x - image_width, 2 * y - image_height]) / min(image_width, image_height)

y_test = model.predict(x_test)
print(y_test.shape)

img_arr = np.zeros((image_width, image_height, 3), dtype=np.float32)
for y in range(0, image_height):
    for x in range(0, image_width):
        img_arr[x][y] = y_test[y * image_width + x]
pyplot.imshow(img_arr)
pyplot.show()
