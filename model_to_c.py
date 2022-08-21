from email.mime import image
import datetime
from keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as pyplot


model = load_model("newmodel.h5")

for layer in model.layers:
    print(layer.weights[0].numpy()) # w
    print("----")
    print(layer.weights[1].numpy()) # b
    print("====")