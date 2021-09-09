from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from load_data import load_train, load_test
from feature import *
import math
import json
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main():
    train_data = load_train(train_path,enable_vis=1 if vis_feature else 0)
    test_data = load_test(test_path,vis_class=vis_feature)
    print('加载完成,开始lda拟合')
    x_train = []
    y_train = []
    key_dic = {"empty": 0}
    i = 0
    for key in train_data:
        if key != "empty":
            i += 1
            key_dic[key] = i
        for single_data in train_data[key]:
            feature_ = get_feature(single_data)
            x_train.append(feature_)
            y_train += [key_dic[key]]
    x_train = np.array(x_train)

    feaature_scaler = StandardScaler()
    x_train = feaature_scaler.fit_transform(x_train)

    # lda_neighbors = math.floor(1.5*min([len(signals) for signals in train_data.values()]))
    # lda = KNeighborsClassifier(n_neighbors=lda_neighbors)
    # lda.fit(x_train, y_train)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(x_train, y_train)

    test_data_feature = get_feature(test_data)
    a = np.repeat([test_data_feature],repeats=test_data["data"].shape[0],axis=0)
    a[:,:test_data["data"].shape[1]]=test_data["data"]
    a = feaature_scaler.transform(a)
    lda_pre = lda.predict(a)

    # lda = PCA(n_components=2)
    # lda.fit(x_train)

    # test_data_=partial_fraction(test_data)
    # test_max_sig_list = test_data_[get_max_index(test_data_, top_k=20)]
    # lda_pre_feature_list = lda.predict(test_max_sig_list)
    lda_pre_feature = lda.predict(feaature_scaler.transform(test_data_feature.reshape(1,-1)))
    pre_result = list(key_dic.keys())[lda_pre_feature[0]]
    signal_index_list = list(range(test_data['data'].shape[0]-20,test_data['data'].shape[0])) #get_max_index(test_data[:,:num_sig], top_k=20)

    print("绘图测试")
    num_sig = test_data['data'].shape[1]
    fig = plt.figure()
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    # lda
    ax1 = fig.add_subplot(121)
    ax1.plot(np.arange(len(lda_pre)), lda_pre, c="b", label="pre")
    ax1.scatter(signal_index_list, lda_pre[signal_index_list], c="r", marker="*")
    ax1.set_ylabel("pre")
    ax1.set_title(label="result:" + pre_result)
    plt.yticks(np.arange(len(key_dic)), list(key_dic.keys()))

    ax1_1 = ax1.twinx()
    test_data_show=StandardScaler().fit_transform(test_data['data'])
    for i in range(num_sig):
        ax1_1.plot(
            np.arange(test_data_show.shape[0]),
            test_data_show[:, i],
            c=colors[i],
            alpha=0.5
        )
    ax1_1.set_ylabel("signal")
    ax1.legend()

    # lda
    ax2 = fig.add_subplot(122)
    for i, key in enumerate(train_data):
        single_list = []
        for single in train_data[key]:

            single_list.append(get_feature(single))
        single_list = feaature_scaler.transform(single_list)
        new_sig = lda.transform(single_list)
        ax2.scatter(new_sig[:, 0], new_sig[:, 1], color=colors[i], alpha=0.5, label=key)

    test_data_lda = lda.transform(a)
    ax2.scatter(test_data_lda[:, 0], test_data_lda[:, 1], color="b", marker=".", s=10)
    ax2.scatter(test_data_lda[signal_index_list, 0], test_data_lda[signal_index_list, 1], color="r", marker=".", s=10, label="target!")
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    train_path = 'train'
    test_path = 'test'

    filename = 'config.json'
    with open(filename, encoding='utf-8') as file_obj:
        names = json.load(file_obj)
    # num_sig = names['num_sig']
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    vis_feature=int(input("""1 A类（透明）：纯净水、雪碧 、(白)醋、酒、糖水
2 B类（茶色）：茶
3 C类（浅黑）：可乐
4 D类（深黑）：咖啡、老抽、生抽
请输入视觉分类结果（1，2，3，4）（直接回车表示不使用视觉信息）：""") or "0")
    print(vis_feature)
    main()
