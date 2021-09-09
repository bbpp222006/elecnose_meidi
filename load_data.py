import numpy as np
import os
from feature import *
import re
import json

filename='config.json'
with open(filename, encoding='utf-8') as file_obj:
    names = json.load(file_obj)
re_name_dic=names['re_name_dic']

num_vis_class = max([item_class[1]for item_class in list(re_name_dic.values())])

def read_file_data(file_path):
    '''
    :param file_path: 文件路径
    :return: {"data":原始数据,"start_index":int 通气下标}
    '''
    return_dic = {}
    data = np.genfromtxt(file_path, delimiter="\t")
    # 删除nan序列
    data = np.delete(data, np.argwhere(np.isnan(data[1, :])), axis=1)
    # 删除时间序列
    # data = np.concatenate((np.arange(data_raw.shape[0]).reshape([-1, 1]), data_raw), axis=1)
    # temp_a = np.corrcoef(data.transpose())
    # timearray_index = np.argwhere(temp_a[0] >= 0.9999)
    # data = np.delete(data, timearray_index, axis=1)
    data = data[:,2:]
    # print("自动检测出时间序列，已删除", timearray_index, end=",")
    nan_flag = data.sum()
    assert np.isnan(nan_flag) == False

    return_dic["data"] = data
    start_index = np.abs(data[:,1]-18).argsort()[::1][0]
    return_dic["start_index"] = start_index
    # assert data.shape[1] == num_sig
    return return_dic


def load_train(raw_data_path,enable_vis=0):
    data_to_save = {}
    for root, dirs, files in os.walk(raw_data_path):
        for file in files:
            for item_regex in re_name_dic:
                if re.search(item_regex, file):
                    item_class = re_name_dic[item_regex][0]
                    break
            file_path = root + "/" + file

            signal_dic = read_file_data(file_path)
            # enable_vis=
            signal_dic["vis"] = enable_vis *(re_name_dic[item_regex][1]+np.random.randn(1)[0]*0.001) #这里要加一点微小的扰动，不然lda拟合会出bug（不清楚原因）
            # raw_signal_samed=partial_fraction(raw_signal)
            # signal = np.concatenate([raw_signal_samed, 0.3*enable_vis*re_name_dic[item_regex][1]*np.ones([raw_signal_samed.shape[0],1])],axis=1)

            # signal = np.concatenate([raw_signal_samed, enable_vis*np.repeat([0.5*np.eye(num_vis_class + 1,
            #                                                              num_vis_class,
            #                                                              k=-1)[re_name_dic[item_regex][1]]],
            #                                                      axis=0,
            #                                                      repeats=raw_signal_samed.shape[0])],axis=1)

            print(file_path, "---->", item_class, signal_dic["data"].shape, ":训练集")
            if not item_class in data_to_save:
                data_to_save[item_class] = [signal_dic]
            else:
                data_to_save[item_class].append(signal_dic)
    return data_to_save


def load_test(test_report, length=1,vis_class=0):
    lists = os.listdir(test_report)  # 列出目录的下所有文件和文件夹保存到lists
    length = min(len(lists), length)
    lists.sort(key=lambda fn: os.path.getmtime(test_report + "/" + fn))  # 按时间排序
    file_new = lists[-length:]  # 获取最新的文件保存到file_new
    file_new = list(map(lambda fn: test_report + "/" + fn, file_new))
    test_data_list = []
    for file_path in file_new:
        signal_dic = read_file_data(file_path)
        # raw_signal_samed = partial_fraction(raw_signal)
        # signal = np.concatenate([raw_signal_samed, np.repeat([0.5*np.eye(num_vis_class+1,
        #                                                              num_vis_class,
        #                                                              k=-1)[vis_class]],
        #                                                      axis=0,
        #                                                      repeats=raw_signal_samed.shape[0])],
        #                         axis=1)
        signal_dic["vis"] = vis_class
        test_data_list.append(signal_dic)
        print(file_path, "---->", signal_dic["data"].shape, ":测试集")
    return test_data_list[0]

# a = read_file_data("test/2.肯亚_1110 110 190.txt")
# print(666)
# train_data=load_train("train")
#
# test_data = load_test("test")
# a=get_max(test_data[0])
# print(test_data[0].shape,a.shape)
