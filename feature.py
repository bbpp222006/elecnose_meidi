import numpy as np


def get_base(signal):
    base = np.mean(signal[0:10, :], axis=0)
    return base


def partial_fraction(signal):
    sig_base = get_base(signal)
    cleaned_sig = difference(signal) / sig_base
    return cleaned_sig


def difference(signal):
    sig_base = get_base(signal)
    cleaned_sig = np.abs(signal - sig_base)
    return cleaned_sig


def log_sig(signal):
    sig_base = get_base(signal)
    cleaned_sig = np.log(np.abs(signal - sig_base))
    return cleaned_sig


def get_max_index(signal, top_k):
    signal_range = np.linalg.norm(signal, axis=1)
    top_k_idx = signal_range.argsort()[::-1][0:top_k]
    return top_k_idx


def get_max(signal, top_k=10):
    # max_index = get_max_index(signal, top_k=top_k)
    # max_index = list(range(signal.shape[0]-10,signal.shape[0]))
    # max_sig_list = signal[max_index]
    # max_feature = np.mean(max_sig_list, axis=0)
    max_feature = np.mean(signal[-10:, :], axis=0)
    return np.array(max_feature)

def get_sig_high_90_idx(signal,start_index,max_feature):

    signal_range = np.linalg.norm(signal[start_index:, :] - 0.9 * max_feature, axis=1)
    sig_high_90_idx = signal_range.argsort()[::1][0:1] + start_index
    return sig_high_90_idx[0]

def get_feature(signal_dic):
    max_feature = get_max(signal_dic['data'])
    vis_feature = signal_dic['vis']
    base_feature = get_base(signal_dic['data'])
    time_feture = get_sig_high_90_idx(signal_dic['data'],signal_dic['start_index'],max_feature)
    sensetive_feature1 = max_feature/base_feature
    sensetive_feature2 = max_feature - base_feature
    feature=np.concatenate([max_feature,base_feature,[time_feture],sensetive_feature1,sensetive_feature2,[vis_feature]])
    return feature
