#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入模块
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn import datasets
import matplotlib.pyplot as plt
# k近邻函数
from sklearn.neighbors import KNeighborsClassifier
# 调用线性回归函数
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC
from sklearn import preprocessing


快速入门 = 0
# https://zhuanlan.zhihu.com/p/33148250
if 0:
    # 建议在下面的python Console中运行，可以实时看到数据详细信息
    print("鸢尾花数据集，用来识别花的种类========")
    iris = datasets.load_iris()
    # Scikit-learn可以从一个或者多个数据集中学习信息，这些数据集合可表示为2维阵列，也可认为是一个列表。
    # 列表的第一个维度代表 样本 ，第二个维度代表 特征 （每一行代表一个样本，每一列代表一种特征）。
    # 这个数据集包含150个样本，每个样本包含4个特征：花萼长度，花萼宽度，花瓣长度，花瓣宽度，详细数据可以通过``iris.DESCR``查看。
    iris_X = iris.data  # 用来分析的数据：shape(150,4)
    iris_y = iris.target  # 对应的分类结果：shape(150,)
    print("花的种类：", np.unique(iris_y))
    # 划分为训练集和测试集数据
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2)
    # 设置knn分类器
    knn = KNeighborsClassifier()
    # 进行训练
    knn.fit(X_train, y_train)
    # 使用训练好的knn进行数据预测
    print("预测值：", knn.predict(X_test))
    print("实际值：", y_test)

    print("波士顿房价数据集==============")
    # 导入数据集
    # 这里将全部数据用于训练，并没有对数据进行划分，上例中将数据划分为训练和测试数据，后面会讲到交叉验证
    loaded_data = datasets.load_boston()
    data_X = loaded_data.data
    data_y = loaded_data.target
    # 设置线性回归模块
    model = LinearRegression()
    # 训练数据，得出参数
    model.fit(data_X, data_y)
    # 利用模型，对新数据，进行预测，与原标签进行比较
    print(model.predict(data_X[:4, :]))
    print(data_y[:4])

深化上面的例子 = 0
# https://www.jianshu.com/p/b5eb165ac2c2
if 1:
    if 0:
        print("knn实现iris的分类,并且使用交叉验证，并且划分成多次进行交叉验证，得到一个准确度列表scores====")
        iris = datasets.load_iris()
        iris_x = iris.data
        iris_y = iris.target
        x_train, x_test, y_train, y_test = train_test_split(iris_x[:100], iris_y[:100], test_size=0.3)
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, iris_x, iris_y, cv=5, scoring='accuracy')  # cross_val_score for classfication
        print(scores)

    if 0:
        print("生成数据，使用linear regression实现回归，画图============")
        # boston = datasets.load_boston()
        # x = boston.data
        # y = boston.target
        x, y = datasets.make_regression(n_samples=50, n_features=1, noise=1)
        x_train, x_test, y_train, y_test = train_test_split(x[:100], y[:100], test_size=0.3)
        linear = LinearRegression()
        linear.fit(x_train, y_train)
        linear.predict(x[:4])
        print(linear.score(x_test, y_test))
        plt.scatter(x, y)
        plt.show()

    if 0:
        print("归一化（正则化）,使用svm进行分类，并画图比较正则化前后的准确率========")
        x, y = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0,
                                            n_informative=1, random_state=22, n_clusters_per_class=1, scale=100)
        x_train, x_test, y_train, y_test = train_test_split(
            x[:100], y[:100], test_size=0.3)
        model = SVC()
        model.fit(x_train, y_train)
        score1 = model.score(x_test, y_test)
        # print(x[:5], y[:5])
        plt.subplot(121)
        plt.scatter(x[:, 0], x[:, 1], c=y)

        x = preprocessing.scale(x)
        y = preprocessing.scale(y)
        x_train, x_test, y_train, y_test = train_test_split(x[:100], y[:100], test_size=0.3)
        model = SVC()
        model.fit(x_train, y_train)
        score2 = model.score(x_test, y_test)
        # print(x[:5], y[:5])
        plt.subplot(122)
        plt.scatter(x[:, 0], x[:, 1], c=y)
        print('precision:', score1)
        print('precision:', score2)
        plt.show()

    if 0:
        print("选择合适的knn参数k，分别在分类、回归===========")
        iris = datasets.load_iris()
        x = iris.data
        y = iris.target
        k_range = range(1, 30)
        k_scores = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            # loss=-cross_val_score(knn,x,y,cv=10,scoring="mean_squared_error")# for regression
            # k_scores.append(loss.mean())
            scores = cross_val_score(
                knn, x, y, cv=10, scoring='accuracy')  # for classification
            k_scores.append(scores.mean())
        plt.plot(k_range, k_scores)
        plt.xlabel('value of k for knn')
        # plt.ylabel('crowss validated loss')
        plt.ylabel('crowss validated accuracy')
        plt.show()

    if 0:
        print("使用validation curve观察学习曲线，此处展示了过拟合的情况==========")
        digits = datasets.load_digits()
        x = digits.data
        y = digits.target
        model = SVC()
        param_range = np.logspace(-6, -2.3, 5)
        train_loss, test_loss = validation_curve(model, x, y, param_name='gamma', param_range=param_range, cv=10,
                                                 scoring='neg_mean_squared_error')  # 数据大小，训练曲线、测试曲线

        train_loss_mean = -np.mean(train_loss, axis=1)
        test_loss_mean = -np.mean(test_loss, axis=1)

        plt.plot(param_range, train_loss_mean, label='train')
        plt.plot(param_range, test_loss_mean, label='cross-validation')
        plt.xlabel('gamma')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.show()

