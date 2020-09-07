import os
import numpy as np

import matplotlib
matplotlib.use('Agg')           # noqa: E402

import tensorflow as tf
import time

import utils
import dataprocess as dp
from myclass import ADV_PAR, TE_PAR, INPUT_PAR

# 指定0号GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_size = 28
img_chan = 1
n_classes = 10

data_set = 'MNIST'

RUNTIME = 30
adv_par = ADV_PAR.ADV_PAR()
fail_name = 'ISTR_MR'
te_par = TE_PAR.TE_PAR(mr_name=['color'])
k_sec = 1000
top_k = 1


class Dummy:    # 空类
    pass


env = Dummy()   # 模型参数
LayerOutput = Dummy()   # 输出值

if data_set == 'MNIST':
    input_par = INPUT_PAR.INPUT_PAR(28, 1, 10, 'MNIST')
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.read_mnist("./MNIST_data/")  # 读数据

elif data_set == 'CIFAR10':
    input_par = INPUT_PAR.INPUT_PAR(32, 3, 10, 'CIFAR10')
    x_train, y_train, x_test, y_test = utils.read_cifar10()

x_train_order, y_train_order, list_order_train = dp.read_order_tr_data('order_data/'+data_set+'/', x_train, y_train)

input_par.input_n_samples(x_train.shape[0])
print('X_train shape = {}'.format(x_train.shape))
print('X_test shape = {}'.format(x_test.shape))

x_train_order, y_train_order = dp.fail_tr_set(fail_name, x_train_order, y_train_order, list_order_train)
x_train, y_train = utils.shuffle_data(x_train_order, y_train_order)

print('\nConstruction graph')
utils.cal_graph(env, input_par)

print('\nInitializing graph')
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())  # local_variables在图中并未被存储的变量

if fail_name == 'ISTR_AE':
    tr_name = fail_name
elif fail_name == 'ISTR_MR':
    tr_name = fail_name + '_' + str(te_par.mr_name) + '_' + str(te_par.mr_k)
else:
    tr_name = fail_name + '_L' + str(te_par.f_label)
print(tr_name)

utils.train(sess, env, x_train, y_train, None, None, epochs=5, batch_size=128,
            load=False, name=tr_name)

print('\nget dict mfr')
time_start = time.time()

dict_mfr = utils.get_dict_mfr_data(sess, env, x_train, y_train, batch_size=128, load_dict_mfr_flag=False)
print("time len = {}".format(time.time() - time_start))

for size in [500, 1000]:
    te_par.te_size = size

    for rate_i in [5]:  # acc_i

        te_par.acc = float(rate_i / 10)

        np.random.seed(0)
        random_list = np.random.choice(range(x_test.shape[0] - size), RUNTIME, replace=False)

        for run_i in range(0, RUNTIME):

            print('\nrate = {}. run_time={}/{}'.format(te_par.acc, run_i+1, RUNTIME))

            print('\ngenerate test set')
            adv_par.edit_par(sess, env, epochs=12, eps=0.02)
            print(te_par.te_size)

            start = random_list[run_i]
            x_test_mr, y_test_mr = dp.acc_te_set(fail_name, data_set, x_test[start:], y_test[start:], adv_par, te_par)

            print('x_test_mr shape = {}, y_test_mr shape={}'.format(x_test_mr.shape, y_test_mr.shape))
            loss, acc = utils.evaluate(sess, env, x_test_mr, y_test_mr, batch_size=128)

            time_start = time.time()
            print('\n'+tr_name)
            print('rate = {}. run_time={}/{}'.format(te_par.acc, run_i + 1, RUNTIME))
            print('acc={}'.format(acc))
            print('te_size={}.'.format(te_par.te_size))
            kmn_cov, nb_cov, sna_cov, _ = utils.neuron_level_cov(sess, env, x_test_mr, y_test_mr, dict_mfr, k_sec)

            print('rate = {}. run_time={}/{}'.format(te_par.acc, run_i + 1, RUNTIME))
            print('acc={}'.format(acc))
            print('te_size={}.'.format(te_par.te_size))
            tkn_cov, tknp_cov = utils.layer_level_cov(sess, env, x_test_mr, y_test_mr, top_k, batch_size=128)
            print("time len = {}".format(time.time() - time_start))

            print('\n'+tr_name)
            print('run_time={}/{}'.format(run_i + 1, RUNTIME))
            print("kmn_cov={}".format(kmn_cov))
            print("nb_cov={}".format(nb_cov))
            print("sna_cov={}".format(sna_cov))
            print("tkn_cov = {}".format(tkn_cov))
            print("tkpn_cov = {}".format(tknp_cov))
            print("time len = {}".format(time.time() - time_start))

            file_path = './cov_res/'+input_par.data_name+'/TE'+str(x_test_mr.shape[0])+'/'+tr_name+'/'
            os.makedirs(file_path, exist_ok=True)
            utils.update_xlsx(file_path+'neuron_cov.xlsx', [rate_i, acc, kmn_cov, nb_cov, sna_cov, tkn_cov, tknp_cov])
            utils.update_xlsx(file_path + 'layer_cov.xlsx', [rate_i, acc, tkn_cov, tknp_cov])
            utils.update_xlsx(file_path+'acc_loss.xlsx', [loss, acc])

        # sess.close()



