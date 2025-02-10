import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Node:
    """决策树节点类"""
    def __init__(self):
        self.feature_index = None  # 分裂特征的索引
        self.split_point = None   # 分裂点
        self.depth = None         # 当前节点深度
        self.left_tree = None     # 左子树
        self.right_tree = None    # 右子树
        self.leaf_class = None    # 叶子节点类别


def gini(y, D):
    """
    计算样本集的加权基尼指数
    :param y: 样本标签数组
    :param D: 样本权重数组
    :return: 加权后的基尼指数
    """
    unique_class = np.unique(y)
    total_weight = np.sum(D)
    gini_index = 1

    for c in unique_class:
        prob = np.sum(D[y == c]) / total_weight
        gini_index -= prob ** 2

    return gini_index


def calc_min_gini_index(a, y, D):
    """
    计算特征a下的最小加权基尼指数及对应的分裂点
    :param a: 单一特征值数组
    :param y: 数据样本标签
    :param D: 样本权重
    :return: 最小基尼指数和对应分裂点
    """
    sorted_feature = np.sort(a)
    total_weight = np.sum(D)

    split_points = [(sorted_feature[i] + sorted_feature[i + 1]) / 2 for i in range(len(sorted_feature) - 1)]
    min_gini = float('inf')
    best_split_point = None

    for point in split_points:
        left_mask = a <= point
        right_mask = a > point

        gini_left = gini(y[left_mask], D[left_mask])
        gini_right = gini(y[right_mask], D[right_mask])

        weighted_gini = (np.sum(D[left_mask]) * gini_left + np.sum(D[right_mask]) * gini_right) / total_weight

        if weighted_gini < min_gini:
            min_gini = weighted_gini
            best_split_point = point

    return min_gini, best_split_point


def choose_feature_to_split(X, y, D):
    """
    选择最佳分裂特征和分裂点
    :param X: 样本特征矩阵
    :param y: 样本标签
    :param D: 样本权重
    :return: 最佳特征索引和分裂点
    """
    best_feature = None
    best_split_point = None
    min_gini = float('inf')

    for feature_index in range(X.shape[1]):
        gini_index, split_point = calc_min_gini_index(X[:, feature_index], y, D)
        if gini_index < min_gini:
            min_gini = gini_index
            best_feature = feature_index
            best_split_point = split_point

    return best_feature, best_split_point


def create_single_tree(X, y, D, depth=0, max_depth=2):
    """
    构建单棵决策树（基学习器）
    :param X: 样本特征矩阵
    :param y: 样本标签
    :param D: 样本权重
    :param depth: 当前树深度
    :param max_depth: 树的最大深度
    :return: 构建的树节点
    """
    node = Node()
    node.depth = depth

    if depth == max_depth or len(X) <= 2:
        pos_weight = np.sum(D[y == 1])
        neg_weight = np.sum(D[y == -1])
        node.leaf_class = 1 if pos_weight > neg_weight else -1
        return node

    feature_index, split_point = choose_feature_to_split(X, y, D)
    node.feature_index = feature_index
    node.split_point = split_point

    left_mask = X[:, feature_index] <= split_point
    right_mask = X[:, feature_index] > split_point

    node.left_tree = create_single_tree(X[left_mask], y[left_mask], D[left_mask], depth + 1, max_depth)
    node.right_tree = create_single_tree(X[right_mask], y[right_mask], D[right_mask], depth + 1, max_depth)

    return node


def predict_single(tree, x):
    """预测单个样本的类别"""
    if tree.leaf_class is not None:
        return tree.leaf_class
    if x[tree.feature_index] > tree.split_point:
        return predict_single(tree.right_tree, x)
    else:
        return predict_single(tree.left_tree, x)


def predict_base(tree, X):
    """基于基学习器预测所有样本的类别"""
    return np.array([predict_single(tree, x) for x in X])


def ada_boost_train(X, y, tree_num=20, max_depth=2):
    """
    训练AdaBoost模型
    :param X: 样本特征矩阵
    :param y: 样本标签
    :param tree_num: 最大基学习器数目
    :param max_depth: 基学习器的最大深度
    :return: 训练好的基学习器列表和对应权重
    """
    D = np.ones(len(y)) / len(y)
    trees = []
    alphas = []

    for _ in range(tree_num):
        tree = create_single_tree(X, y, D, max_depth=max_depth)
        predictions = predict_base(tree, X)
        error_rate = np.sum(D[predictions != y])

        if error_rate == 0 or error_rate > 0.5:
            break

        alpha = 0.5 * np.log((1 - error_rate) / max(error_rate, 1e-16))
        alphas.append(alpha)
        trees.append(tree)

        D *= np.exp(-alpha * y * predictions)
        D /= np.sum(D)

    return trees, alphas


def ada_boost_predict(X, trees, alphas):
    """使用AdaBoost模型预测样本类别"""
    agg_estimates = sum(alpha * predict_base(tree, X) for tree, alpha in zip(trees, alphas))
    return np.where(agg_estimates >= 0, 1, -1)


def plot_ada_boost_decision_bound(X, y, trees, alphas):
    """绘制AdaBoost分类器的决策边界"""
    pos = y == 1
    neg = y == -1
    x_range = np.linspace(0, 1, 600)
    y_range = np.linspace(-0.2, 0.7, 600)
    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)

    Z = ada_boost_predict(np.c_[X_mesh.ravel(), Y_mesh.ravel()], trees, alphas).reshape(X_mesh.shape)
    plt.contour(X_mesh, Y_mesh, Z, levels=[0], colors='orange', linewidths=1)
    plt.scatter(X[pos, 0], X[pos, 1], label='Positive', color='c')
    plt.scatter(X[neg, 0], X[neg, 1], label='Negative', color='lightcoral')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_path = r'.\data\watermelon3_0a_Ch.txt'
    data = pd.read_table(data_path, delimiter=' ')

    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values
    y[y == 0] = -1

    trees, alphas = ada_boost_train(X, y)
    plot_ada_boost_decision_bound(X, y, trees, alphas)
