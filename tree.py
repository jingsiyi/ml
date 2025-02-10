import pandas as pd
import numpy as np
from math import log2
from graphviz import Digraph

# 读取数据
data = pd.read_csv("watermelon_dataset.csv")

# 定义全局变量，用于节点编号
node_id = 0

# 计算信息熵
def calculate_entropy(data):
    class_counts = data['好瓜'].value_counts()
    probabilities = class_counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# 计算信息增益
def calculate_information_gain(data, feature):
    total_entropy = calculate_entropy(data)
    feature_values = data[feature].unique()
    weighted_entropy = 0
    for value in feature_values:
        subset = data[data[feature] == value]
        subset_entropy = calculate_entropy(subset)
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    info_gain = total_entropy - weighted_entropy
    return info_gain

# 选择最佳特征
def select_best_feature(data):
    features = [col for col in data.columns if col not in ['好瓜', '编号']]
    best_info_gain = -1
    best_feature = None
    for feature in features:
        info_gain = calculate_information_gain(data, feature)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    return best_feature

# 构建决策树，返回树结构和节点信息
def build_decision_tree(data, dot=None, parent=None, edge_label=''):
    global node_id
    if dot is None:
        dot = Digraph()
        dot.node(name=str(node_id), label='开始')
        parent = str(node_id)
        node_id += 1

    # 如果数据集中的所有实例都属于同一类，返回该类
    if len(data['好瓜'].unique()) == 1:
        label = data['好瓜'].iloc[0]
        dot.node(name=str(node_id), label=label, shape='ellipse', style='filled', color='lightgrey')
        dot.edge(parent, str(node_id), label=edge_label)
        node_id += 1
        return dot

    # 如果没有更多特征可以划分，返回出现次数最多的类
    if len(data.columns) == 2:  # 只有目标类和编号
        label = data['好瓜'].mode()['好瓜'][0]
        dot.node(name=str(node_id), label=label, shape='ellipse', style='filled', color='lightgrey')
        dot.edge(parent, str(node_id), label=edge_label)
        node_id += 1
        return dot

    # 选择最佳划分特征
    best_feature = select_best_feature(data)
    dot.node(name=str(node_id), label=best_feature)
    if parent is not None:
        dot.edge(parent, str(node_id), label=edge_label)
    current_node = str(node_id)
    node_id += 1

    # 对每个特征值递归构建子树
    feature_values = data[best_feature].unique()
    for value in feature_values:
        subset = data[data[best_feature] == value].drop(columns=[best_feature])
        dot = build_decision_tree(subset, dot, current_node, str(value))
    return dot

# 构建并可视化决策树
dot = build_decision_tree(data)
dot.render('decision_tree', format='png', cleanup=True)
