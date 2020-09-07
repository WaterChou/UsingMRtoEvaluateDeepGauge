import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import mr
import math

x_train, y_train, x_valid, y_valid, x_test, y_test = utils.read_mnist("./MNIST_data/")  # 读数据

os.makedirs('./mr_img/', exist_ok=True)

img = x_train[0]
# mr_img = mr.img_affine(img, 15)
mr_img = mr.img_elastic(np.reshape(img, (28, 28, -1)), alpha=50, sigma=4.5)
# plt.imsave('./mr_img/ae.png', np.reshape(mr_img, (28, 28)), cmap='gray')
# mr_img = mr.img_rotation(img, 10)
# mr_img = mr.img_transition(img, -5, -5, 'nearest')
# mr_img = mr.img_color(mr_img, 1, 0.9)
# mr_img = mr.img_pepper_noise(mr_img, 0.2)
# mr_img = mr.img_affine(img, 15)
# mr_img = mr.img_random_noise(img, 0.2)
# mr_img = mr.img_shear(img, -15, -15)
# mr_img = mr.img_scaling(img, 1, 1.5)
# plt.imsave('./mr_img/e_10.png', np.reshape(mr_img_tmp, (28, 28)), cmap='gray')
plt.imsave('./mr_img/e_50_4p5.png', np.reshape(mr_img, (28, 28)), cmap='gray')
np.random.seed(0)
for i in range(5):

    random_list = np.random.choice(range(x_test.shape[0] - 1000), 5, replace=False)
    print(random_list)
