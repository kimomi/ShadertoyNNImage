from keras.models import load_model

model = load_model("newmodel.h5")

result = ""
layer_num = 0
print(model.layers.__len__())
for layer in model.layers:
    ws = layer.weights[0].numpy()
    bs = layer.weights[1].numpy()

    (input_num, output_num) = layer.weights[0].numpy().shape
    for i in range(0, output_num):
        x = "x_{0}_{1}".format(layer_num + 1, i)
        result += "float " + x + " = "
        for j in range(0, input_num):
            w = ws[j][i]
            result += "{0} * x_{1}_{2} + ".format(w, layer_num, j)
        b = bs[i]
        result += str(b) + ";\n"

        if model.layers.__len__() > layer_num + 1:
            result += "{0} = Relu({0});\n".format(x)
        else:
            result += "{0} = Sigmoid({0});\n".format(x)
    layer_num += 1

f = open("gencode.c", "w+")
f.write(result)
f.close()
print("gen code to gencode.c")
