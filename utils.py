import os
import numpy as np
import json
import pickle
import openpyxl as opxl
from collections import defaultdict
from copy import deepcopy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from cnn_model import lenet5_model, vgg16_model
from attacks import fast_gradient


def read_mnist(readpath):
    """"
    读数据
    """
    print('\nLoading MNIST')
    img_size = 28
    img_chan = 1

    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)

    mnist = input_data.read_data_sets(readpath, one_hot=True)
    X_train = mnist.train.images
    X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
    y_train = mnist.train.labels

    X_valid = mnist.validation.images
    X_valid = np.reshape(X_valid, [-1, img_size, img_size, img_chan])
    y_valid = mnist.validation.labels

    X_test = mnist.test.images
    X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
    y_test = mnist.test.labels

    tf.logging.set_verbosity(old_v)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def shuffle_data(x_data, y_data):

    ind = np.arange(x_data.shape[0])
    np.random.shuffle(ind)
    x_data = x_data[ind]
    y_data = y_data[ind]

    return x_data, y_data


def unpickle(file_path):

    with open(file_path, 'rb') as fo:
        # data_dict = pickle.load(fo, encoding='bytes')  # official
        data_dict = pickle.load(fo, encoding='latin1')

    return data_dict


def read_cifar10_label_name(file_path):

    data_dict = unpickle(file_path)
    # dict_keys(['num_vis', 'label_names', 'num_cases_per_batch'])

    label_name = data_dict['label_names']  # len=10

    return label_name


def read_cifar10_train(train_path):

    for i in range(5):
        data_dict = unpickle(train_path[i])
        # dict_keys(['filenames', 'batch_label', 'data', 'labels'])
        x_data = data_dict['data']   # shape(10000, 3072)
        y_data = np.array(data_dict['labels'])  # shape(10000,)
        x_data = x_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # shape(10000, 32, 32, 3)

        if i == 0:
            x_train = x_data
            y_train = y_data
        else:
            x_train = np.concatenate((x_train, x_data), axis=0)
            y_train = np.concatenate((y_train, y_data), axis=0)

    return x_train, y_train


def read_cifar10_test(test_path):

    data_dict = unpickle(test_path)
    x_test = data_dict['data']  # shape(10000, 3072)
    y_test = np.array(data_dict['labels'])  # shape(10000,)
    x_test = x_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # shape(10000, 32, 32, 3)

    return x_test, y_test


def label_dense_to_one_hot(label_dense, class_num):

    n_sample = label_dense.shape[0]
    index_offset = np.arange(n_sample) * class_num  # 长度为n_sample的1D等差数组，起始值为0，步长为class_num
    label_one_hot = np.zeros(shape=(n_sample, class_num))
    label_one_hot.flat[index_offset + label_dense.ravel()] = 1  # flat:迭代器

    return label_one_hot


def read_cifar10():

    cifar10_label_name_path = file = './CIFAR10_data/batches.meta'
    cifar10_test_data_path = './CIFAR10_data/test_batch'

    cifar10_train_data_path = ['./CIFAR10_data/data_batch_1',
                               './CIFAR10_data/data_batch_2',
                               './CIFAR10_data/data_batch_3',
                               './CIFAR10_data/data_batch_4',
                               './CIFAR10_data/data_batch_5']

    print('\nLoading CIFAR-10')

    x_train, y_train = read_cifar10_train(cifar10_train_data_path)
    x_test, y_test = read_cifar10_test(cifar10_test_data_path)
    label_name_list = read_cifar10_label_name(cifar10_label_name_path)
    # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(label_name_list)

    y_train = label_dense_to_one_hot(y_train, 10)
    y_test = label_dense_to_one_hot(y_test, 10)

    return x_train, y_train, x_test, y_test


def get_fgsm_data(sess, env, X_test, eps=0.02, epochs=12, load_adv_flag=False, save_adv_flag=True):

    if load_adv_flag:
        print('Load adv data')
        X_adv = np.load("adv_data/fgsm.npy")
        print(np.shape(X_adv))
    else:
        print('Generating adversarial data')
        X_adv = make_fgsm(sess, env, X_test, epochs, eps, 64)

        if save_adv_flag:
            print('Save adv data')
            os.makedirs('adv_data', exist_ok=True)
            np.save("adv_data/fgsm.npy", X_adv)

    return X_adv


def get_dict_mfr_data(sess, env, X_train, Y_train, batch_size=64, file_name="dict_mfr.json",
                      load_dict_mfr_flag=False, save_dict_mfr_flag=True):

    if load_dict_mfr_flag is True:
        print("Load " + file_name + " ...")
        f = open("./dict/"+file_name, 'r')
        dict_mfr = json.load(f)
        f.close()
    else:
        dict_mfr = get_dict_mfr(sess, env, X_train, Y_train, batch_size)

        if save_dict_mfr_flag is True:
            os.makedirs('dict', exist_ok=True)
            f = open("./dict/"+file_name, 'w', encoding='utf-8')
            f.write(json.dumps(dict_mfr))
            f.close()
            print("Save " + file_name + " successfully.")

    return dict_mfr


def save_dict_file(file_name, dict_file):

    os.makedirs('dict', exist_ok=True)
    f = open("./dict/" + file_name, 'w', encoding='utf-8')
    f.write(json.dumps(dict_file))
    f.close()
    print("Save " + file_name + " successfully.")


def write_xlsx(file_path, list_input_data):

    wb = opxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(list_input_data)
    wb.save(file_path)


def update_xlsx(file_path, list_input_data):

    if os.path.exists(file_path):
        wb = opxl.load_workbook(file_path)
        ws = wb['Sheet1']
        ws.append(list_input_data)
        wb.save(file_path)
    else:
        write_xlsx(file_path, list_input_data)


def cal_graph(env, input_par):

    img_size = input_par.img_size
    img_chan = input_par.img_chan
    n_classes = input_par.n_class
    data_name = input_par.data_name

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan), name='x')
        env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
        env.training = tf.placeholder_with_default(False, (), name='mode')  # 训练模式/非训练模式
        env.lr = tf.placeholder(tf.float32)

        if data_name == 'MNIST':
            env.ybar, logits, env.layer_dict = lenet5_model.LeNet5model(env.x, True, logits=True, training=env.training)
        elif data_name == 'CIFAR10':
            VGG16model = vgg16_model.VGG16(env.x, 0.5, n_classes, env.training)
            logits, env.ybar = VGG16model.output

        with tf.variable_scope('acc'):
            count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        with tf.variable_scope('loss'):
            if data_name == 'MNIST':
                xent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=env.y, logits=logits)  # xent是一个向量
                env.loss = tf.reduce_mean(xent, name='loss')
            elif data_name == 'CIFAR10':
                regularizer = tf.contrib.layers.l2_regularizer(scale=0.5 / input_par.n_sample)
                reg_term = tf.contrib.layers.apply_regularization(regularizer)
                xent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=env.y, logits=logits)  # xent是一个向量
                env.loss = tf.reduce_mean(xent, name='loss') + reg_term

        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer(learning_rate=env.lr)
            env.train_op = optimizer.minimize(env.loss)

        env.saver = tf.train.Saver()  # 保存模型,默认保存全部变量
    '''
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        env.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
        env.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
        env.x_fgsm = fast_gradient.fgm(lenet5_model.LeNet5model, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)
    '''


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):   # 判断对象是否包含对应的属性
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model_save/{}/{}'.format(name, name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    print(X_data.shape)
    n_batch = int((n_sample+batch_size-1) / batch_size)
    lr = 1e-3
    train_acc = 0
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            # print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)

            if train_acc >= 0.99:
                lr = lr*0.1

            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True,
                                              env.lr: lr})
        print('\nEvaluate on train set')
        _, train_acc = evaluate(sess, env, X_data, y_data)

        if X_valid is not None:
            print('\nEvaluate on valid set')
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\nSaving model')
        os.makedirs('model_save/'+name, exist_ok=True)
        env.saver.save(sess, 'model_save/{}/{}'.format(name, name))


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]  # shape[0]：第一维长度
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_fgsm(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgsm, feed_dict={env.x: X_data[start:end],
                                              env.fgsm_eps: eps,
                                              env.fgsm_epochs: epochs})
        X_adv[start:end] = adv

    return X_adv


def update_dict_mfr(dict_mfr_tmp, dict_mfr):

    layer_name = list(dict_mfr_tmp.keys())

    for i in range(len(layer_name)):
        neuron_num = len(dict_mfr_tmp[layer_name[i]])

        for j in range(neuron_num):
            dict_mfr[layer_name[i]][str(j)][0] = min(dict_mfr_tmp[layer_name[i]][str(j)][0],
                                                     dict_mfr[layer_name[i]][str(j)][0])
            dict_mfr[layer_name[i]][str(j)][1] = max(dict_mfr_tmp[layer_name[i]][str(j)][1],
                                                     dict_mfr[layer_name[i]][str(j)][1])

    return dict_mfr


def cmp_dict_neuron(dict1, dict2):
    layer_name = list(dict1.keys())

    for i in range(len(layer_name)):
        neuron_num = len(dict1[layer_name[i]])
        for j in range(neuron_num):
            if dict1[layer_name[i]][j] == dict2[layer_name[i]][j]:
                continue
            else:
                return False

    return True


def cmp_dict_mfr(dict1, dict2):
    layer_name = list(dict1.keys())

    for i in range(len(layer_name)):
        neuron_num = len(dict1[layer_name[i]])
        for j in range(neuron_num):
            if dict1[layer_name[i]][str(j)] == dict2[layer_name[i]][str(j)]:
                continue
            else:
                return False
    return True


def get_dict_sample_batch(sess, env, X_data, Y_data, start, end, flag_batch, flag_acc=0):

    dict_sample_batch = defaultdict(dict)

    list_neuron_num = []  # 每层神经元数量
    layer_name = []

    cnt = end - start

    if flag_acc:
        acc_batch = sess.run(env.acc, feed_dict={env.x: X_data[start:end], env.y: Y_data[start:end]})

    for i in range(cnt):
        dict_sample_batch[i] = defaultdict(dict)
        X_tmp = np.expand_dims(X_data[start + i], 0)
        dict_sample_batch[i] = sess.run(env.layer_dict, feed_dict={env.x: X_tmp})

        if i == 0:
            layer_name = list(dict_sample_batch[0].keys())

        for j in range(len(layer_name)):
            dict_sample_batch[i][layer_name[j]] = dict_sample_batch[i][layer_name[j]].flatten()
            if i == 0:
                list_neuron_num.append(np.shape(dict_sample_batch[0][layer_name[j]])[0])

        for j in range(len(layer_name)):
            dict_tmp = deepcopy(dict_sample_batch[i][layer_name[j]])
            dict_sample_batch[i][layer_name[j]] = defaultdict(float)

            for k in range(list_neuron_num[j]):
                dict_sample_batch[i][layer_name[j]].setdefault(k, float(dict_tmp[k]))

    if flag_batch == 0:
        if flag_acc:
            return dict_sample_batch, layer_name, list_neuron_num, acc_batch
        else:
            return dict_sample_batch, layer_name, list_neuron_num
    else:
        if flag_acc:
            return dict_sample_batch, acc_batch
        else:
            return dict_sample_batch


def get_dict_neuron_batch(dict_sample_batch, layer_name, list_neuron_num, cnt):

    dict_neuron_batch = defaultdict(dict)

    for i in range(len(layer_name)):
        dict_neuron_batch[layer_name[i]] = defaultdict(list)

        for j in range(list_neuron_num[i]):
            for k in range(cnt):
                dict_neuron_batch[layer_name[i]][j].append(dict_sample_batch[k][layer_name[i]][j])

    return dict_neuron_batch


def get_dict_mfr(sess, env, X_data, Y_data, batch_size=128):

    """
    dict_sample = {
        0: {    # sample_num
            'layer_name': {
                0: 0.0    # float: neuron_num
            }
        }
    }
    dict_neuron = {
        'layer_name': {
            0: [0.0, 0.0, ...,0.0]  # list: neuron_value, len(list)=n_sample
        }
    }
    dict_mfr={
        'layer_name':{
            0: [lown, highn]    # list: neuron major function region on all samples
        }
    }
    """

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)

    list_neuron_num = []    # 每层神经元数量
    layer_name = []

    for batch in range(n_batch):
        print('batch: {0}/{1}'.format(batch+1, n_batch))
        # print('\nGet dict_sample_batch information...')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end-start

        if batch == 0:
            dict_sample_batch, layer_name, list_neuron_num = get_dict_sample_batch(sess, env, X_data, Y_data,
                                                                                   start, end, batch)
        else:
            dict_sample_batch = get_dict_sample_batch(sess, env, X_data, Y_data, start, end, batch)

        # print('\nGet dict_neuron_batch information...')

        dict_neuron_batch = get_dict_neuron_batch(dict_sample_batch, layer_name, list_neuron_num, cnt)

        print('Get dict_mfr_batch information...')

        dict_mfr_batch = defaultdict(dict)

        for i in range(len(layer_name)):
            dict_mfr_batch[layer_name[i]] = defaultdict(list)

            for j in range(list_neuron_num[i]):
                dict_mfr_batch[layer_name[i]][str(j)].append(min(dict_neuron_batch[layer_name[i]][j]))
                dict_mfr_batch[layer_name[i]][str(j)].append(max(dict_neuron_batch[layer_name[i]][j]))

        if dict_mfr_batch is False:
            print('error: dict_mfr_batch is empty')
        else:
            if batch == 0:
                dict_mfr = deepcopy(dict_mfr_batch)
            else:
                print('Update dict_mfr information...')
                dict_mfr = update_dict_mfr(dict_mfr_batch, dict_mfr)

    return dict_mfr


def ith_section_covered(neuron_value, list_lh, k_sec):

    lh_range = list_lh[1] - list_lh[0]
    sec_range = float(lh_range/k_sec)

    list_k_sec = []

    list_k_sec.append(list_lh[0])
    for i in range(1, k_sec):
        list_k_sec.append(list_k_sec[-1]+sec_range)
    list_k_sec.append(list_lh[1])   # len=1001

    if neuron_value < list_lh[0] or neuron_value > list_lh[1]:
        if neuron_value < list_lh[0]:
            return -2   # lower corner
        else:
            return -1   # upper corner
    else:
        for i in range(len(list_k_sec)):
            if i == k_sec:
                return 0
            else:
                if neuron_value <= list_k_sec[i+1]:
                    return i+1


def init_dict_cover(layer_name, list_neuron_num, k_sec):

    dict_sec_cover = defaultdict(dict)
    dict_upboundary_cover = defaultdict(dict)
    dict_lowboundary_cover = defaultdict(dict)

    for i in range(len(layer_name)):
        dict_upboundary_cover[layer_name[i]] = defaultdict(int)
        dict_lowboundary_cover[layer_name[i]] = defaultdict(int)
        dict_sec_cover[layer_name[i]] = defaultdict(list)

        for j in range(list_neuron_num[i]):
            for k in range(k_sec):
                dict_sec_cover[layer_name[i]][j].append(0)  # 初始化

    return dict_sec_cover, dict_upboundary_cover, dict_lowboundary_cover


def get_dict_cover_batch(dict_sample_batch, dict_mfr, layer_name, list_neuron_num, k_sec, cnt):

    dict_sec_cover_batch, dict_upboundary_cover_batch, dict_lowboundary_cover_batch = \
        init_dict_cover(layer_name, list_neuron_num, k_sec)

    for i in range(cnt):
        for j in range(len(layer_name)):
            for k in range(list_neuron_num[j]):
                neuron_value = dict_sample_batch[i][layer_name[j]][k]
                list_lh = dict_mfr[layer_name[j]][str(k)]
                sec_flag = ith_section_covered(neuron_value, list_lh, k_sec)

                if sec_flag != 0:
                    if sec_flag > 0:
                        dict_sec_cover_batch[layer_name[j]][k][sec_flag-1] = 1
                    else:
                        if sec_flag < 0:
                            if sec_flag == -1:  # upper corner
                                dict_upboundary_cover_batch[layer_name[j]][k] = 1
                            else:  # lower corner
                                dict_lowboundary_cover_batch[layer_name[j]][k] = 1
                else:
                    print('error: sec_flag = 0')

    return dict_sec_cover_batch, dict_upboundary_cover_batch, dict_lowboundary_cover_batch


def update_dict_cover(dict_cover_tmp, dict_cover, layer_name, list_neuron_num, k_sec):

    for i in range(len(layer_name)):
        neuron_num = list_neuron_num[i]

        for j in range(neuron_num):
            if k_sec:   # update dict_sec_cover
                for k in range(k_sec):
                    if dict_cover_tmp[layer_name[i]][j][k]:
                        dict_cover[layer_name[i]][j][k] = dict_cover_tmp[layer_name[i]][j][k]
            else:
                if dict_cover_tmp[layer_name[i]][j]:
                    dict_cover[layer_name[i]][j] = dict_cover_tmp[layer_name[i]][j]

    return dict_cover


def update_dict_sec_num(dict_sec_tmp, dict_sec, layer_name, list_neuron_num, iter_num):

    for i in range(len(layer_name)):
        neuron_num = list_neuron_num[i]

        for j in range(neuron_num):
            dict_sec[layer_name[i]][str(j)] =\
                float(dict_sec[layer_name[i]][str(j)]*(iter_num-1)+dict_sec_tmp[layer_name[i]][str(j)])/iter_num

    return dict_sec


def cal_coverage_neuron(list_neuron_sec_cover, k_sec):

    KMNCov_up = sum(list_neuron_sec_cover)
    KMNCov_neuron = float(KMNCov_up / k_sec)

    return KMNCov_neuron


def cal_coverage_model(dict_sec_cover, dict_upboundary_cover, dict_lowboundary_cover,
                       layer_name, list_neuron_num, k_sec, flag_return_sec=False):

    KMNCov, UCN_num, LCN_num = 0, 0, 0
    all_neuron_number = sum(list_neuron_num)
    dict_sec_num = defaultdict(dict)

    for i in range(len(layer_name)):
        neuron_number = list_neuron_num[i]
        dict_sec_num[layer_name[i]] = defaultdict(int)

        for j in range(neuron_number):
            dict_sec_num[layer_name[i]][str(j)] = sum(dict_sec_cover[layer_name[i]][j])
            KMNCov += cal_coverage_neuron(dict_sec_cover[layer_name[i]][j], k_sec)
            UCN_num += dict_upboundary_cover[layer_name[i]][j]
            LCN_num += dict_lowboundary_cover[layer_name[i]][j]

    KMNCov = float(KMNCov / all_neuron_number)
    NBCov = float((UCN_num + LCN_num) / (2 * all_neuron_number))
    SNACov = float(UCN_num / all_neuron_number)

    if flag_return_sec:
        return KMNCov, NBCov, SNACov, dict_sec_num
    else:
        return KMNCov, NBCov, SNACov


def neuron_level_cov(sess, env, X_data, Y_data, dict_mfr, k_sec, batch_size=128):

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)

    dict_sec_cover = defaultdict(dict)
    dict_upboundary_cover = defaultdict(dict)
    dict_lowboundary_cover = defaultdict(dict)

    acc = 0.0

    for batch in range(n_batch):
        print('\nbatch: {0}/{1}'.format(batch + 1, n_batch))
        print('Get dict_sample_batch information...')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start

        if batch == 0:
            dict_sample_batch, layer_name, list_neuron_num, acc_batch = get_dict_sample_batch(sess, env, X_data, Y_data,
                                                                                              start, end, batch, 1)
        else:
            dict_sample_batch, acc_batch = get_dict_sample_batch(sess, env, X_data, Y_data, start, end, batch, 1)

        dict_sec_cover_batch, dict_upboundary_cover_batch, dict_lowboundary_cover_batch = \
            get_dict_cover_batch(dict_sample_batch, dict_mfr, layer_name, list_neuron_num, k_sec, cnt)

        acc_batch = float(acc_batch)
        acc += acc_batch * cnt

        if dict_upboundary_cover_batch is False or dict_upboundary_cover_batch is False or dict_sec_cover_batch is False:
            print('error: dict_cover_batch is empty')
        else:
            print('Calculate batch coverage...')

            # KMNCov_batch, NBCov_batch, SNACov_batch = cal_coverage_model(dict_sec_cover_batch,
            #                                                              dict_upboundary_cover_batch,
            #                                                              dict_lowboundary_cover_batch,
            #                                                              layer_name, list_neuron_num, k_sec)
            #
            # print('KMNCov_batch = {0}'.format(KMNCov_batch))
            # print('NBCov_batch = {0}'.format(NBCov_batch))
            # print('SNACov_batch = {0}'.format(SNACov_batch))
            # print('acc_batch = {0}'.format(acc_batch))

            if batch == 0:
                dict_sec_cover = deepcopy(dict_sec_cover_batch)
                dict_upboundary_cover = deepcopy(dict_upboundary_cover_batch)
                dict_lowboundary_cover = deepcopy(dict_lowboundary_cover_batch)
            else:
                print('Updating coverage dict....')
                dict_sec_cover = update_dict_cover(dict_sec_cover_batch, dict_sec_cover,
                                                   layer_name, list_neuron_num, k_sec)
                dict_lowboundary_cover = update_dict_cover(dict_lowboundary_cover_batch, dict_lowboundary_cover,
                                                           layer_name, list_neuron_num, k_sec=0)
                dict_upboundary_cover = update_dict_cover(dict_upboundary_cover_batch, dict_upboundary_cover,
                                                          layer_name, list_neuron_num, k_sec=0)

    print('Calculate coverage...')

    KMNCov, NBCov, SNACov, dict_sec_num = cal_coverage_model(dict_sec_cover, dict_upboundary_cover,
                                                             dict_lowboundary_cover,
                                                             layer_name, list_neuron_num, k_sec, True)
    return KMNCov, NBCov, SNACov, acc/n_sample


def get_dict_topk_batch(dict_sample_batch, layer_name, list_neuron_num, cnt, top_k):

    dict_topk_batch = defaultdict(dict)
    list_layer_value = []

    for i in range(len(layer_name)):
        dict_topk_batch[layer_name[i]] = defaultdict(int)

        for j in range(list_neuron_num[i]):

            for k in range(cnt):
                list_layer_value = list(dict_sample_batch[k][layer_name[i]].values())
                list_layer_value.sort(reverse=True)  # 降序
                top_k_value = list_layer_value[top_k-1]

                if dict_sample_batch[k][layer_name[i]][j] >= top_k_value:
                    dict_topk_batch[layer_name[i]][j] += 1

    return dict_topk_batch


def layer_level_cov(sess, env, x_data, y_data, top_k, batch_size=128):

    if top_k <= 0:
        print('error: top_k <= 0')
        return False

    n_sample = x_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)

    list_neuron_num = []  # 每层神经元数量
    layer_name = []

    for batch in range(n_batch):
        print('\nbatch: {0}/{1}'.format(batch + 1, n_batch))
        print('\nGet dict_sample_batch information...')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start

        if batch == 0:
            dict_sample_batch, layer_name, list_neuron_num = get_dict_sample_batch(sess, env, x_data, y_data,
                                                                                   start, end, batch)
        else:
            dict_sample_batch = get_dict_sample_batch(sess, env, x_data, y_data,
                                                      start, end, batch)

        print('Get dict_topk_batch information...')
        dict_topk_batch = get_dict_topk_batch(dict_sample_batch, layer_name, list_neuron_num, cnt, top_k)

        if dict_topk_batch is False:
            print('error: dict_topk_batch is empty')
        else:
            if batch == 0:
                dict_topk = deepcopy(dict_topk_batch)
            else:
                print('Update dict_topk information...')
                dict_topk = update_dict_cover(dict_topk_batch, dict_topk, layer_name, list_neuron_num, k_sec=0)

    TKNCov, TKNPat = 0, 0
    all_neuron_number = sum(list_neuron_num)

    for i in range(len(layer_name)):
        for j in range(list_neuron_num[i]):
            if dict_topk[layer_name[i]][j]:
                TKNCov += 1
                TKNPat += dict_topk[layer_name[i]][j]

    TKNCov = float(TKNCov/all_neuron_number)
    TKNPat = TKNPat / top_k

    return TKNCov, TKNPat
