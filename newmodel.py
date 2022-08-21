# 将噪声显示为图像
import math
from os import path
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import msvcrt
from keras.models import load_model
import matplotlib.pyplot as pyplot
import datetime

img = np.array(Image.open("reference_middle.jpg"))
image_width = img.shape[0]
image_height = img.shape[1]
image_channel = img.shape[2]
print("image shape, width:" + str(image_width) + ",height:" + str(image_height))
print("load image finish...")

# 范围重新映射为 (-1, 1)
x_train = np.zeros(shape=(image_height*image_width, 2), dtype=np.float32)
for y in range(0, image_height):
    for x in range(0, image_width):
        x_train[y * image_width + x] = np.array([2 * x - image_width, 2 * y - image_height]) / min(image_width, image_height)

y_train = np.zeros(shape=(image_height*image_width, 3), dtype=np.float32)
for y in range(0, image_height):
    for x in range(0, image_width):
        c = img[x, y] / 255
        # y_train[y * image_width + x] = 0.11 * c[0] + 0.59 * c[1] + 0.3 * c[2]
        y_train[y * image_width + x] = c

print("set data finish..., x shape:" + x_train.shape.__str__() + ", y shape:" + y_train.shape.__str__())

if not path.exists("newmodel.h5"):
    model = keras.Sequential([
        tf.keras.layers.Dense(16, activation = 'relu', input_shape = (2,)),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dense(3, activation = 'sigmoid')
        ])
    print("set model finish...")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss= tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()]
        )
    print("compile model finish...")
else:
    model = load_model("newmodel.h5")
    print("load model finish...")

while True:
    if msvcrt.kbhit():
        if msvcrt.getwche() == '\r':
            break
    print("trian model ing...") 
    model.fit(x_train, y_train, epochs=32)
    t = "newmodel" + str(int(np.floor(time.time())))
    model.save(t + ".h5")
    # 绘制一下当前的图片
    y_test = model.predict(x_train)
    img_arr = np.zeros((image_width, image_height, 3), dtype=np.float32)
    for y in range(0, image_height):
        for x in range(0, image_width):
            img_arr[x][y] = y_test[y * image_width + x]
    pyplot.imshow(img_arr)
    pyplot.savefig("images/" + t + ".png")
    pyplot.close()

print("trian model finish...")

model.save("newmodel.h5")

# img_arr = np.zeros((image_width, image_height, 3), dtype=np.float32)
# for y in range(0, image_height):
#     for x in range(0, image_width):
#         img_arr[x][y] = y_train[y * image_width + x]
#         # img_arr[y][x][0] = noise(x + 0.5)
#         # img_arr[y][x][1] = noise(y + 0.5)
#         # img_arr[y][x][2] = 0

# pyplot.imshow(img_arr)
# pyplot.show()

