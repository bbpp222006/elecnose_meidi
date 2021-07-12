from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from load_data import load_train, load_test
from feature import *
import math


def main():
    train_data = load_train(train_path, num_sig=num_sig,enable_vis=1 if vis_feature else 0)
    test_data = load_test(test_path, num_sig=num_sig,vis_class=vis_feature)
    print('加载完成,开始knn拟合')
    x_train = []
    y_train = []
    key_dic = {"empty": 0}
    i = 0
    for key in train_data:
        if key != "empty":
            i += 1
            key_dic[key] = i
        for single_data in train_data[key]:
            feature_ = get_feature(single_data,num_sig=num_sig)
            x_train.append(feature_)
            y_train += [key_dic[key]]
    x_train = np.array(x_train)

    knn_neighbors = math.ceil(1.5*min([len(signals) for signals in train_data.values()]))
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
    knn.fit(x_train, y_train)
    knn_pre = knn.predict(test_data)

    pca = PCA(n_components=2)
    pca.fit(x_train)

    # test_data_=partial_fraction(test_data)
    # test_max_sig_list = test_data_[get_max_index(test_data_, top_k=20)]
    # knn_pre_feature_list = knn.predict(test_max_sig_list)
    knn_pre_feature = knn.predict([get_feature(test_data,num_sig=num_sig)])
    pre_result = list(key_dic.keys())[knn_pre_feature[0]]
    signal_index_list = get_max_index(test_data[:,:num_sig], top_k=20)

    print("绘图测试")
    fig = plt.figure()
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    # knn
    ax1 = fig.add_subplot(121)
    ax1.plot(np.arange(len(knn_pre)), knn_pre, c="b", label="pre")
    ax1.scatter(signal_index_list, knn_pre[signal_index_list], c="r", marker="*")
    ax1.set_ylabel("pre")
    ax1.set_title(label="result:" + pre_result)
    plt.yticks(np.arange(len(key_dic)), list(key_dic.keys()))

    ax1_1 = ax1.twinx()
    test_data_show=test_data
    for i in range(num_sig):
        ax1_1.plot(
            np.arange(test_data_show.shape[0]),
            test_data_show[:, i],
            c=colors[i],
            alpha=0.5
        )
    ax1_1.set_ylabel("signal")
    ax1.legend()

    # pca
    ax2 = fig.add_subplot(122)
    for i, key in enumerate(train_data):
        single_list = []
        for single in train_data[key]:
            single_list.append(get_feature(single,num_sig=num_sig))
        new_sig = pca.transform(single_list)
        ax2.scatter(new_sig[:, 0], new_sig[:, 1], color=colors[i], alpha=0.5, label=key)

    test_data_pca = pca.transform(test_data_show)
    ax2.scatter(test_data_pca[:, 0], test_data_pca[:, 1], color="b", marker=".", s=10)
    ax2.scatter(test_data_pca[signal_index_list, 0], test_data_pca[signal_index_list, 1], color="r", marker=".", s=10, label="target!")
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    train_path = 'train'
    test_path = 'test'
    num_sig = 8  # 传感器个数
    vis_feature=int(input("""1 A类（透明）：纯净水、雪碧 、(白)醋、酒、糖水
2 B类（茶色）：茶
3 C类（浅黑）：可乐
4 D类（深黑）：咖啡、老抽、生抽
请输入视觉分类结果（1，2，3，4）（直接回车表示不使用视觉信息）：""") or "0")
    print(vis_feature)
    main()
