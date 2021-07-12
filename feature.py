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
    max_index = get_max_index(signal, top_k=top_k)
    max_sig_list = signal[max_index]
    max_feature = np.mean(max_sig_list, axis=0)
    return np.array(max_feature)


def get_feature(signal,num_sig):
    max_feature = get_max(signal[:, :num_sig])
    vis_feature = signal[0,num_sig:]
    feature=np.concatenate([max_feature,vis_feature])
    return feature
