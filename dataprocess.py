import numpy as np
import os
import math
import utils
import mr


def order_dataset(x_data, y_data):

    n_class = y_data.shape[1]
    n_sample = y_data.shape[0]
    list_index = np.zeros(shape=(n_class,), dtype=np.uint16)
    x_order = np.zeros(shape=(1, x_data.shape[1], x_data.shape[2], x_data.shape[3]))
    y_order = np.zeros(shape=(1, y_data.shape[1]))

    for sample_i in range(n_sample):
        label = np.argmax(y_data[sample_i])

        x_order = np.insert(x_order, list_index[label], x_data[sample_i], axis=0)
        y_order = np.insert(y_order, list_index[label], y_data[sample_i], axis=0)

        if label < (n_class-1):
            list_index[label+1:] += 1

        print("i={}, label= {}".format(sample_i, label))
        print(list_index)

    x_order = np.delete(x_order, -1, axis=0)
    y_order = np.delete(y_order, -1, axis=0)

    return x_order, y_order, list_index


def sub_order_dataset(x_order, y_order, list_index, n_per_label):

    n_label = len(list_index)
    x_order_sub = np.zeros(shape=(1, x_order.shape[1], x_order.shape[2], x_order.shape[3]))
    y_order_sub = np.zeros(shape=(1, y_order.shape[1]))
    list_index_sub = np.zeros(shape=(n_label,), dtype=np.uint16)

    for i in range(n_label):

        start = list_index[i]
        end = start+n_per_label
        x_order_sub = np.concatenate((x_order_sub, x_order[start: end]), axis=0)
        y_order_sub = np.concatenate((y_order_sub, y_order[start: end]), axis=0)
        list_index_sub[i] = n_per_label*i

    x_order_sub = np.delete(x_order_sub, 0, axis=0)
    y_order_sub = np.delete(y_order_sub, 0, axis=0)

    return x_order_sub, y_order_sub, list_index_sub


def read_order_tr_data(file_path, x_train, y_train):

    if os.path.exists(file_path + 'x_train_order.npy'):
        x_train_order = np.load(file_path + 'x_train_order.npy')
        y_train_order = np.load(file_path + 'y_train_order.npy')
        list_order_train = np.load(file_path + 'list_order_train.npy')

    else:
        x_train_order, y_train_order, list_order_train = order_dataset(x_train, y_train)
        os.makedirs(file_path, exist_ok=True)
        np.save(file_path + 'x_train_order.npy', x_train_order)
        np.save(file_path + 'y_train_order.npy', y_train_order)
        np.save(file_path + 'list_order_train.npy', list_order_train)

    return x_train_order, y_train_order,  list_order_train


def read_order_te_data(file_path, x_test, y_test):

    if os.path.exists(file_path + 'x_test_order.npy'):
        x_test_order = np.load(file_path + 'x_test_order.npy')
        y_test_order = np.load(file_path + 'y_test_order.npy')
        list_order_test = np.load(file_path + 'list_order_test.npy')
    else:
        x_test_order, y_test_order, list_order_test = order_dataset(x_test, y_test)
        os.makedirs(file_path, exist_ok=True)
        np.save(file_path + 'x_test_order.npy', x_test_order)
        np.save(file_path + 'y_test_order.npy', y_test_order)
        np.save(file_path + 'list_order_test.npy', list_order_test)

    return x_test_order, y_test_order,list_order_test


def pollute_sample(x_data, pol_p, pol_s):

    n_sample = x_data.shape[0]

    for sample_i in range(n_sample):

        x_data[sample_i, pol_p: pol_p+pol_s, pol_p: pol_p+pol_s, :] = 1   # 黑底（0），白字（1）

    return x_data


def ptr_set(x_data_order, list_order_index, label, rate):

    n_class = len(list_order_index)
    n_sample = x_data_order.shape[0]

    if label < (n_class-1):
        n_sample_ptr = int(rate*(list_order_index[label+1]-list_order_index[label]))
    else:
        n_sample_ptr = int(rate*(n_sample-list_order_index[label]))

    start = list_order_index[label]
    print(n_sample_ptr)

    for sample_i in range(start, start+n_sample_ptr):

        ptr_p = 0

        x_data_order[sample_i, ptr_p: ptr_p+5, ptr_p: ptr_p+5, :] = 1   # 黑底（0），白字（1）

    return x_data_order


def acc_ptr_set(x_order, y_order, list_index, te_par):

    n_class = len(list_index)

    acc = te_par.acc
    te_size = te_par.te_size
    label = te_par.f_label

    n_pass = int(acc*te_size)
    n_pass_per = int(n_pass/(n_class-1))
    n_fail = te_size-n_pass

    start = list_index[label]
    end = start+n_fail

    x_data, y_data = x_order[start: end], y_order[start: end]

    res_i = 1
    for label_i in range(n_class):

        if label_i != label:
            start = list_index[label_i]
            if res_i < (n_class-1):
                end = start+n_pass_per
            else:
                end = start+n_pass-(res_i-1)*n_pass_per
            res_i += 1

            x_data = np.concatenate((x_data, x_order[start: end]), axis=0)
            y_data = np.concatenate((y_data, y_order[start: end]), axis=0)

    return x_data, y_data


def ibtr_set(x_order, y_order, list_order_index, label, rate):

    n_class = len(list_order_index)
    n_sample = x_order.shape[0]

    if label < (n_class - 1):
        n_sample_ibtr = int(rate * (list_order_index[label + 1] - list_order_index[label]))
    else:
        n_sample_ibtr = int(rate * (n_sample - list_order_index[label]))

    start = list_order_index[label]
    for i in range(n_sample_ibtr):

        x_order = np.delete(x_order, start+i, axis=0)
        y_order = np.delete(y_order, start+i, axis=0)

        if label < (n_class - 1):
            list_order_index[label+1:] -= 1

    return x_order, y_order


def acc_ibtr_set(x_order, y_order, list_index, te_par):

    n_class = len(list_index)

    acc = te_par.acc
    te_size = te_par.te_size
    label = te_par.f_label

    n_pass = int(acc * te_size)
    n_pass_per = int(n_pass / (n_class - 1))
    n_fail = te_size - n_pass

    start = list_index[label]
    end = start + n_fail

    x_data, y_data = x_order[start: end], y_order[start: end]

    res_i = 1
    for label_i in range(n_class):

        if label_i != label:
            start = list_index[label_i]
            if res_i < (n_class - 1):
                end = start + n_pass_per
            else:
                end = start + n_pass - (res_i - 1) * n_pass_per
            res_i += 1

            x_data = np.concatenate((x_data, x_order[start: end]), axis=0)
            y_data = np.concatenate((y_data, y_order[start: end]), axis=0)

    return x_data, y_data


def acc_istr_ae_set(x_org, y_org, x_adv, y_adv, te_par):

    acc = te_par.acc
    te_size = te_par.te_size

    n_pass = int(acc * te_size)
    n_fail = te_size - n_pass

    x_data, y_data = x_org[:n_pass], y_org[:n_pass]

    x_data = np.concatenate((x_data, x_adv[:n_fail]), axis=0)
    y_data = np.concatenate((y_data, y_adv[:n_fail]), axis=0)

    return x_data, y_data


def a_mr_set(x_data, y_data, te_par):

    acc = te_par.acc
    te_size = te_par.te_size
    mr_name = te_par.mr_name[0]

    n_org = int(te_size * acc)
    n_mr = te_size - n_org
    print("n_org {}".format(n_org))

    x_out = x_data[:n_org]
    y_out = y_data[:n_org]

    if n_mr > 0:

        x_mr, y_mr = mr.mr_output(x_data[:n_mr], y_data[:n_mr], mr_name, k=1)

        x_out = np.concatenate((x_out, x_mr), axis=0)
        y_out = np.concatenate((y_out, y_mr), axis=0)

    print(x_out.shape)

    return x_out, y_out


def multi_mr_set(x_data, y_data, te_par):

    te_size = te_par.te_size
    mr_name = te_par.mr_name
    n_mr = te_par.n_mr
    multi_mr = te_par.multi_mr

    if multi_mr == 'C':

        n_source = math.ceil(te_size/(1+n_mr))  # 向上取整
        n_follow = te_size-n_source

        x_out, y_out = x_data[:n_source], y_data[:n_source]

        for mr_i in range(n_mr):

            x_tmp, y_tmp = mr.mr_output(x_data[:n_follow], y_data[:n_follow], mr_name[mr_i], 1)

            x_out = np.concatenate((x_out, x_tmp), axis=0)
            y_out = np.concatenate((y_out, y_tmp), axis=0)

        return x_out, y_out

    elif multi_mr == 'D':

        n_follow = math.floor(te_size/2)    # 向下取整
        n_per_mr = math.floor(n_follow/n_mr)
        n_source = te_size-n_follow

        x_out, y_out = x_data[:n_source], y_data[:n_source]

        for mr_i in range(n_mr):
            start = mr_i*n_per_mr
            end = min(start+n_per_mr, n_follow)

            x_tmp, y_tmp = mr.mr_output(x_data[start:end], y_data[start:end], mr_name[mr_i], 1)

            x_out = np.concatenate((x_out, x_tmp), axis=0)
            y_out = np.concatenate((y_out, y_tmp), axis=0)

        return x_out, y_out


def acc_istr_mr_set(x_test, y_test, te_par):

    if te_par.multi_mr == 'N':

        return a_mr_set(x_test, y_test, te_par)

    else:
        return multi_mr_set(x_test, y_test, te_par)


def fail_tr_set(fail_name, x_order, y_order, list_order_index, label=0, rate=1.0):

    if fail_name == 'PTR':
        x_data = ptr_set(x_order, list_order_index, label, rate)
        y_data = y_order

    elif fail_name == 'IBTR':

        x_data, y_data = ibtr_set(x_order, y_order, list_order_index, label, rate)

    elif fail_name.find('ISTR') != -1:

        x_data, y_data = x_order, y_order

    else:

        print("\nfail name not found")
        return None, None

    return x_data, y_data


def acc_te_set(fail_name, data_set, x_test, y_test, adv_par, te_par):

    if fail_name == 'PTR':
        x_order, y_order, list_index = read_order_te_data('order_data/'+data_set+'/', x_test, y_test)

        x_data, y_data = acc_ptr_set(x_order, y_order, list_index, te_par)

    elif fail_name == 'IBTR':
        x_order, y_order, list_index = read_order_te_data('order_data/' + data_set + '/', x_test, y_test)
        x_data, y_data = acc_ibtr_set(x_order, y_order, list_index, te_par)

    elif fail_name == 'ISTR_AE':

        x_adv = utils.make_fgsm(adv_par.sess, adv_par.env, x_test,
                                adv_par.epochs, adv_par.eps, adv_par.batch_size)
        y_adv = y_test

        x_data, y_data = acc_istr_ae_set(x_test, y_test, x_adv, y_adv, te_par)

    elif fail_name == 'ISTR_MR':

        x_data, y_data = acc_istr_mr_set(x_test, y_test, te_par)

    else:

        print("\nfail name not found")
        return None, None

    return x_data, y_data


# def ictr_set(x_order, y_order, list_order_index, label, rate):






