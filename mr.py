import numpy as np
import random
import scipy.ndimage as scimg_nd
import skimage.exposure as skimg_exp
import skimage.util as skimg_util
import cv2
import math


def img_flip(img, mode):

    new_img = np.zeros_like(img, dtype=np.float)

    if mode == 'horizon':
        new_img = np.flip(img, axis=1)

    elif mode == 'vertical':
        new_img = np.flip(img, axis=0)

    return new_img


def img_transition(img, shift_r, shift_c, mode):

    # mode = ‘constant’, ‘nearest’
    new_img = scimg_nd.shift(img, (shift_r, shift_c, 0), mode=mode)

    return new_img


def img_scaling(img, scaling_r, scaling_c, mode='edge'):

    img_r = img.shape[0]
    img_c = img.shape[1]

    # scaling = random.uniform(0.5, 1.5)

    r = int(img_r * scaling_r)
    c = int(img_c * scaling_c)

    img_tmp = cv2.resize(img, (c, r))

    if r < img_r:
        up = int(abs(r - img_r) / 2)
        img_tmp = np.pad(img_tmp, ((up, img_r - r - up), (0, 0)), mode=mode)
        start_r = 0
    else:
        start_r = int((r-img_r) / 2)

    if c < img_c:
        left = int(abs(c - img_c) / 2)
        img_tmp = np.pad(img_tmp, ((0, 0), (left, img_c - c - left)), mode=mode)
        start_c = 0
    else:
        start_c = int((c - img_c) / 2)

    new_img = img_tmp[start_r:start_r + img_r, start_c:start_c + img_c]

    return new_img


def img_rotation(img, ang):

    img_r = img.shape[0]
    img_c = img.shape[1]

    matrix = cv2.getRotationMatrix2D((img_c / 2, img_r / 2), ang, 1)

    new_img = cv2.warpAffine(img, matrix, (img_r, img_c))

    return new_img


def img_pepper_noise(img, theta):   # theta <= 10

    # noise = np.random.uniform(0, float(theta/10), size=img.shape)
    #
    # new_img = img+noise

    img_r = img.shape[0]
    img_c = img.shape[1]

    size = img_r*img_c

    new_img = img.copy()

    n_pepper = int(theta*size)

    for i in range(n_pepper):

        randx = np.random.randint(1, img_r - 1)  # 生成一个 1 至 img_r-1 之间的随机整数
        randy = np.random.randint(1, img_c - 1)  # 生成一个 1 至 img_c-1 之间的随机整数

        if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
            new_img[randx, randy] = 0
        else:
            new_img[randx, randy] = 1

    return new_img


def img_gaussian_noise(img, theta):

    return skimg_util.random_noise(img, mode='gaussian', var=theta, clip=True)


def img_poisson_noise(img):

    return skimg_util.random_noise(img, mode='poisson', clip=True)


def img_multiplicative_noise(img, theta):

    return skimg_util.random_noise(img, mode='speckle', var=theta, clip=True)


def img_random_erasing(img, mask_r, mask_c, mode):

    img_r = img.shape[0]
    img_c = img.shape[1]

    # mask_r = round(img_r*random.uniform(0.0, 0.5))
    # mask_c = round(img_c*random.uniform(0.0, 0.5))

    mask_x = random.randint(0, img_c-mask_c-1)
    mask_y = random.randint(0, img_r-mask_r-1)

    new_img = np.zeros_like(img, dtype=np.float)

    if mode == 'noise':
        mask = np.random.randint(0, 30, size=(mask_r, mask_c, img.shape[-1]))
    elif mode == 'constant':
        mask = -1 * img[mask_y:mask_y + +mask_r, mask_x:mask_x + mask_c, :]

    new_img[mask_y:mask_y + mask_r, mask_x:mask_x + mask_c, :] = mask

    new_img = img + new_img

    return new_img


def img_color(img, theta, alpha):

    # new_img = abs(img-np.ones_like(img)*theta)

    new_img = abs(theta * img - np.ones_like(img)*alpha)

    return new_img


def img_random_line(img, theta):

    line = np.ones(shape=(img.shape[-2], img.shape[-1]))*abs(1-img[0][0][0])

    n_line = int(img.shape[0]*theta)
    new_img = img

    for i in range(n_line):
        new_img[random.randint(0, img.shape[0]-1)] = line

    return new_img


def img_shear(img, angle_x, angle_y):

    angle_x = math.pi*angle_x / 180.0
    angle_y = math.pi * angle_y / 180.0

    shape = img.shape
    shape_size = shape[:2]

    M_shear = np.array([[1, np.tan(angle_x), 0],
                        [np.tan(angle_y), 1, 0]], dtype=np.float32)
    # print(M_shear.shape)

    return cv2.warpAffine(img, M_shear, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # cv2.BORDER_REFLECT_101


def img_affine(img, alpha_affine):

    random_state = np.random.RandomState(None)

    shape = img.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]+square_size, center_square[1]-square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # pts1 – Coordinates of triangle vertices in the source image.
    # pts2 – Coordinates of the corresponding triangle vertices in the destination image.
    affine_matrix = cv2.getAffineTransform(pts1, pts2)  # (2,3)
    new_img = cv2.warpAffine(img, affine_matrix, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT)

    return new_img


def img_elastic(image, alpha, sigma, random_state=None):

    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = scimg_nd.filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = scimg_nd.filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    # dx_tmp = (random_state.rand(*shape) * 2 - 1) * alpha
    # dy_tmp = (random_state.rand(*shape) * 2 - 1) * alpha
    # dz_tmp = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    # indices_tmp = np.reshape(y + dy_tmp, (-1, 1)), np.reshape(x + dx_tmp, (-1, 1)), np.reshape(z, (-1, 1))

    new_img = scimg_nd.map_coordinates(np.reshape(image, shape), indices, order=1, mode='reflect').reshape(shape)
    # new_img_tmp = scimg_nd.map_coordinates(np.reshape(image, shape), indices_tmp, order=1, mode='reflect').reshape(shape)

    # return new_img, new_img_tm
    return new_img


def mr_rotation(x_data, y_data, k):

    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.float)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.float)

    for i in range(k):
        print('{0}/{1}'.format(i + 1, k))

        for j in range(x_data.shape[0]):
            print('{0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')
            angle = random.randint(10, 45)
            img = np.reshape(img_rotation(x_data[j], angle), (-1, img_r, img_c, img_cha))
            x_mr = np.concatenate((x_mr, img), axis=0)
            y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)

    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    return x_mr, y_mr


def mr_scalling(x_data, y_data, k):

    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.float)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.float)

    for i in range(k):
        print('{0}/{1}'.format(i + 1, k))

        for j in range(x_data.shape[0]):
            print('{0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')

            flag = True
            while flag:
                scaling_r = random.uniform(0.5, 1.5)
                scaling_c = random.uniform(0.5, 1.5)

                if (scaling_c == 1) and (scaling_r == 1):
                    flag = True
                else:
                    flag = False

            img = np.reshape(img_scaling(x_data[j], scaling_r, scaling_c), (-1, img_r, img_c, img_cha))
            x_mr = np.concatenate((x_mr, img), axis=0)
            y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)

    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    return x_mr, y_mr


def mr_zoomin(x_data, y_data, k):   # smaller

    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.float)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.float)

    for i in range(k):
        print('{0}/{1}'.format(i + 1, k))

        for j in range(x_data.shape[0]):
            print('{0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')
            scaling = random.uniform(1, 1.5)
            img = np.reshape(img_scaling(x_data[j], scaling, scaling), (-1, img_r, img_c, img_cha))
            x_mr = np.concatenate((x_mr, img), axis=0)
            y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)

    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    return x_mr, y_mr


def mr_zoomout(x_data, y_data, k):  # larger

    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.float)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.float)

    for i in range(k):
        print('{0}/{1}'.format(i + 1, k))

        for j in range(x_data.shape[0]):
            print('{0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')
            scaling = random.uniform(0.5, 1)
            img = np.reshape(img_scaling(x_data[j], scaling, scaling), (-1, img_r, img_c, img_cha))
            x_mr = np.concatenate((x_mr, img), axis=0)
            y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)

    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    return x_mr, y_mr


def mr_flip(x_data, y_data, k, flag_dir):

    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.float)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.float)

    mode_list = ['horizon', 'vertical']

    for i in range(k):
        print('k {0}/{1}'.format(i + 1, k))

        for j in range(x_data.shape[0]):
            print(' {0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')
            img = np.reshape(img_flip(x_data[j], mode_list[flag_dir]), (-1, img_r, img_c, img_cha))
            x_mr = np.concatenate((x_mr, img), axis=0)
            y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)

    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    return x_mr, y_mr


def mr_transition(x_data, y_data, k):

    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.float)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.float)

    for i in range(k):
        print('k {0}/{1}'.format(i + 1, k))

        j = 0
        while j < x_data.shape[0]:
            shift_r = random.randint(-int(img_r * 0.2), int(img_r * 0.2))
            shift_c = random.randint(-int(img_c * 0.2), int(img_c * 0.2))

            if shift_c + shift_r:
                print(' {0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')
                img = img_transition(x_data[j], shift_r, shift_c, 'nearest')
                img = np.reshape(img, (-1, img_r, img_c, img_cha))
                x_mr = np.concatenate((x_mr, img), axis=0)
                y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)
                j = j+1

    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    return x_mr, y_mr


def mr_pepper_noise(x_data, y_data, k):

    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.float)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.float)

    for i in range(k):
        print('{0}/{1}'.format(i + 1, k))

        for j in range(x_data.shape[0]):
            print('{0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')
            theta = random.randint(10, 20)
            img = img_pepper_noise(x_data[j], float(theta/100))
            img = np.reshape(img, (-1, img_r, img_c, img_cha))
            x_mr = np.concatenate((x_mr, img), axis=0)
            y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)

    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    return x_mr, y_mr


def mr_color(x_data, y_data, k):

    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.float)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.float)

    for i in range(k):
        print('{0}/{1}'.format(i + 1, k))

        for j in range(x_data.shape[0]):
            print('{0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')
            alpha = random.uniform(0.2, 0.9)
            img = img_color(x_data[j], 1, alpha)
            img = np.reshape(img, (-1, img_r, img_c, img_cha))
            x_mr = np.concatenate((x_mr, img), axis=0)
            y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)

    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    return x_mr, y_mr


def mr_random_line(x_data, y_data, k):

    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.float)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.float)

    for i in range(k):
        print('{0}/{1}'.format(i + 1, k))

        for j in range(x_data.shape[0]):
            print('{0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')
            theta = float(random.randint(10, 30)/100)
            img = img_random_line(x_data[j], 1+theta)
            img = np.reshape(img, (-1, img_r, img_c, img_cha))
            x_mr = np.concatenate((x_mr, img), axis=0)
            y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)

    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    return x_mr, y_mr


def mr_shear(x_data, y_data, k):

    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.float)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.float)

    for i in range(k):
        print('{0}/{1}'.format(i + 1, k))

        for j in range(x_data.shape[0]):
            print('{0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')

            flag = True
            while flag:
                theta_x = random.randrange(-20, 20)  # [a,b]
                theta_y = random.randrange(-20, 20)
                if (abs(theta_x) < 10) and (abs(theta_y) < 10):
                    flag = True
                else:
                    flag = False

            img = img_shear(x_data[j], theta_x, theta_y)
            img = np.reshape(img, (-1, img_r, img_c, img_cha))
            x_mr = np.concatenate((x_mr, img), axis=0)
            y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)

    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    return x_mr, y_mr


def mr_affine(x_data, y_data, k):

    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.float)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.float)

    for i in range(k):
        print('{0}/{1}'.format(i + 1, k))

        for j in range(x_data.shape[0]):
            print('{0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')
            theta = random.randint(8, 12)
            img = img_affine(x_data[j], theta)
            img = np.reshape(img, (-1, img_r, img_c, img_cha))
            x_mr = np.concatenate((x_mr, img), axis=0)
            y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)

    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    return x_mr, y_mr


def mr_elastic(x_data, y_data, k):

    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.float)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.float)

    for i in range(k):
        print('{0}/{1}'.format(i + 1, k))

        for j in range(x_data.shape[0]):
            print('{0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')
            elastic_alpha = random.randint(75, 80)
            img = img_elastic(x_data[j], elastic_alpha, sigma=4.5)
            img = np.reshape(img, (-1, img_r, img_c, img_cha))
            x_mr = np.concatenate((x_mr, img), axis=0)
            y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)

    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    return x_mr, y_mr


def mr_output(x_data, y_data, mr_name, k):

    if mr_name == 'flip_h':

        return mr_flip(x_data, y_data, k, 0)

    elif mr_name == 'transition':

        return mr_transition(x_data, y_data, k)

    elif mr_name == 'rotation':

        return mr_rotation(x_data, y_data, k)

    elif mr_name == 'color':

        return mr_color(x_data, y_data, k)

    elif mr_name == 'pepper':

        return mr_pepper_noise(x_data, y_data, k)

    elif mr_name =='frtcl':

        x_tmp, y_tmp = mr_flip(x_data, y_data, k, flag_dir=0)
        x_tmp, y_tmp = mr_rotation(x_tmp, y_tmp, 1)
        x_tmp, y_tmp = mr_transition(x_tmp, y_tmp, 1)
        x_tmp, y_tmp = mr_color(x_tmp, y_tmp, 1)

        return mr_random_line(x_tmp, y_tmp, 1)

    elif mr_name == 'scaling':

        return mr_scalling(x_data, y_data, k)

    elif mr_name == 'shear':

        return mr_shear(x_data, y_data, k)

    elif mr_name == 'affine':

        return mr_affine(x_data, y_data, k)

    elif mr_name == 'elastic':

        return mr_elastic(x_data, y_data, k)

    elif mr_name == 'ecl':

        x_tmp, y_tmp = mr_elastic(x_data, y_data, k)
        x_tmp, y_tmp = mr_color(x_tmp, y_tmp, 1)

        return mr_random_line(x_tmp, y_tmp, 1)

    print('\nMR not found')

    return 0, 0


