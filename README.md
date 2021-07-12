# elecnose_meidi

#### 介绍
**多通道电子鼻识别算法**

此项目使用KNN与PCA算法进行多维时间序列识别

#### 更新记录
v2_2更新：
1. 修复不使用视觉信息时的pca绘图错误

v2_1更新：
1. 优化图像特征信息表示方式（onehot编码），现在pca显示效果更加均匀

v2_0更新：
1. 新增图像识别接口，可选是否加入视觉信息

v1_2更新：
1. 新增knn超参数自动判断，现在可以尝试不同数量的训练集了
2. 优化画图显示界面

---------------------------------
#### 常见问题以及解决方法
>为什么我的main.py文件打不开（打开报错）？
* 之前版本的python没有卸载干净，或者安装版本出错，重装应该可以解决。

>为什么程序运行（main.py）后出错？
* 出现这种情况有很多可能，请先尝试以下解决方法，若仍有报错，这种情况请附上报错信息发起issue。issue
格式参考[提问的智慧](https://github.com/ryanhanwu/How-To-Ask-Questions-The-Smart-Way)

1. **检查项目文件夹路径是否含有中文（最好是纯英文目录）。**
2. 软件安装过程环境是否配好，漏掉步骤等。
3. 重新下载本项目文件，并尝试运行。


#### 安装教程
1. python安装

    [下载地址](https://www.python.org/)  
    **注意安装时勾选上path**  

2. python的各种依赖包

    现成算法只用到了
    1. sklearn
    2. matplotlib
    
    接下来是操作流程
    按下win+r，输入cmd打开命令行窗口  
    复制粘贴以下内容   
    **逐行输入，等安装完了再输下一条**
    ```
    pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

#### 使用说明

train文件夹：存放采集到的数据，用于训练  

test文件夹：存放待测试数据，**每次只留一个文件！**

main：程序入口，包含文件加载、绘图等流程

feature：信号特征提取，数据处理等函数

load_data：加载数据、归一化、基线处理等

#### 参与贡献

1.  Fork 本仓库
2.  新建分支
3.  提交代码
4.  新建 Pull Request


#### 须知

1.  本项目中的train、test文件夹内为测试样本，只具备参考价值
