from keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as pyplot

model = load_model("newmodel.h5")

image_width = 512
image_height = 256
image_channel = 3

# remap to (-1, 1)
x_test = np.zeros(shape=(image_height*image_width, 2), dtype=np.float32)
for y in range(0, image_height):
    for x in range(0, image_width):
        x_test[y * image_width + x] = np.array([2 * x - image_width, 2 * y - image_height]) / min(image_width, image_height)
y_test = model.predict(x_test)

# get image out put
img_arr = np.zeros((image_height, image_width, 3), dtype=np.float32)
for y in range(0, image_height):
    for x in range(0, image_width):
        img_arr[y][x] = y_test[y * image_width + x]
pyplot.imshow(img_arr)
pyplot.show()
