from scipy import misc
import mnist_loader
import numpy as np

traning_data, validation_data, test_data = mnist_loader.load_data_wrapper()
traning_data.remove(traning_data[0])

t = traning_data[0][0] + traning_data[1][0]


print (type(traning_data[0]))

a = mnist_loader.load_extend_fusion_mnist_loader()


#traning_data.remove(traning_data[0])

#print(image_array)

image_array = t

image_array = image_array.reshape(28, 28)

filename = 'mnist_train.jpg'


misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)
