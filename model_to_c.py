from keras.models import load_model
import numpy as np


model = load_model("newmodel.h5")

result = ""
layer_num = 0
for layer in model.layers:
    # print(layer)
    # print(layer.weights[0].numpy().shape) # w
    (input_num, output_num) = layer.weights[0].numpy().shape
    for i in range(0, output_num):
        x = "x_" + str(layer_num + 1) + "_" + str(i)
        result += "float " + x + " = "
        for j in range(0, input_num):
            w = layer.weights[0].numpy()[j][i]
            result += str(w) + " * x_" + str(layer_num) + "_" + str(j) + " + "
        b = layer.weights[1].numpy()[i]
        result += str(b) + ";\n"
        result += x + " = Relu(" + x + ")" + ";\n"
    layer_num += 1
    # print("----")
    # print(layer.weights[1].numpy()) # b
    # print("====")

f = open("gencode.c", "w+")
f.write(result)
f.close()

print("gen code to gencode.c")