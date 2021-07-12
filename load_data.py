import numpy as np
import os
from feature import *
import re

re_name_dic = {
    '空': ['empty',1],
    "茶": ["tea",2],
    "咖啡": ["coffee",4],
    "雪碧": ["spirit",1],
    "可乐": ["cola",3],
    "白醋": ["white_vinegar",1],
    "老抽": ["dark_soysauce",4],
    "生抽": ["soy_sauce",4],
    "酒": ["wine",1],
    "糖水": ["sugar",1],
}

num_vis_class = max([item_class[1]for item_class in list(re_name_dic.values())])

def read_file_data(file_path, num_sig=8):
    data = np.genfromtxt(file_path, delimiter='	')
    nan_flag = data.sum()
    if np.isnan(nan_flag):
        print('已删除nan', end=",")
        data = data[:, :-1]
    if data.shape[1] > num_sig:
        print('已删除时间序列', end=",")
        data = data[:, -num_sig:]
    nan_flag = data.sum()
    assert np.isnan(nan_flag) == False
    assert data.shape[1] == num_sig
    return data


def load_train(raw_data_path, num_sig=8,enable_vis=0):
    data_to_save = {}
    for root, dirs, files in os.walk(raw_data_path):
        for file in files:
            for item_regex in re_name_dic:
                if re.search(item_regex, file):
                    item_class = re_name_dic[item_regex][0]
                    break
            file_path = root + "/" + file

            raw_signal = read_file_data(file_path, num_sig=num_sig)
            raw_signal_samed=partial_fraction(raw_signal)
            # signal = np.concatenate([raw_signal_samed, 0.3*enable_vis*re_name_dic[item_regex][1]*np.ones([raw_signal_samed.shape[0],1])],axis=1)

            signal = np.concatenate([raw_signal_samed, enable_vis*np.repeat([0.5*np.eye(num_vis_class + 1,
                                                                         num_vis_class,
                                                                         k=-1)[re_name_dic[item_regex][1]]],
                                                                 axis=0,
                                                                 repeats=raw_signal_samed.shape[0])],axis=1)

            print(file_path, "---->", item_class, signal.shape, ":训练集")
            if not item_class in data_to_save:
                data_to_save[item_class] = [signal]
            else:
                data_to_save[item_class].append(signal)
    return data_to_save


def load_test(test_report, length=1, num_sig=8,vis_class=0):
    lists = os.listdir(test_report)  # 列出目录的下所有文件和文件夹保存到lists
    length = min(len(lists), length)
    lists.sort(key=lambda fn: os.path.getmtime(test_report + "/" + fn))  # 按时间排序
    file_new = lists[-length:]  # 获取最新的文件保存到file_new
    file_new = list(map(lambda fn: test_report + "/" + fn, file_new))
    test_data_list = []
    for file_path in file_new:
        raw_signal = read_file_data(file_path, num_sig=num_sig)
        raw_signal_samed = partial_fraction(raw_signal)
        signal = np.concatenate([raw_signal_samed, np.repeat([0.5*np.eye(num_vis_class+1,
                                                                     num_vis_class,
                                                                     k=-1)[vis_class]],
                                                             axis=0,
                                                             repeats=raw_signal_samed.shape[0])],
                                axis=1)

        test_data_list.append(signal)
        print(file_path, "---->", signal.shape, ":测试集")
    return test_data_list[0]

# train_data=load_train("train")
#
# test_data = load_test("test")
# a=get_max(test_data[0])
# print(test_data[0].shape,a.shape)
