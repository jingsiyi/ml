from libsvm.svmutil import *


data_path = "watermelon_dataset.libsvm"
y, X = svm_read_problem(data_path)


param_linear = '-t 0 -c 100'#10000
param_rbf = '-t 2 -g 0.5 -c 100'#10000


model_linear = svm_train(y, X, param_linear)
p_label_linear, p_acc_linear, p_val_linear = svm_predict(y, X, model_linear)
linear_sv_count = model_linear.get_nr_sv()
linear_accuracy = p_acc_linear[0]


model_rbf = svm_train(y, X, param_rbf)
p_label_rbf, p_acc_rbf, p_val_rbf = svm_predict(y, X, model_rbf)
rbf_sv_count = model_rbf.get_nr_sv()
rbf_accuracy = p_acc_rbf[0]

linear_sv_indices = model_linear.get_sv_indices()
rbf_sv_indices = model_rbf.get_sv_indices()

linear_sv = set(linear_sv_indices)
rbf_sv = set(rbf_sv_indices)
common_sv = linear_sv.intersection(rbf_sv)
unique_to_linear = linear_sv - rbf_sv
unique_to_rbf = rbf_sv - linear_sv

print("线性核支持向量数量:", linear_sv_count)
print("高斯核支持向量数量:", rbf_sv_count)
print("线性核模型的训练精度: {:.2f}%".format(linear_accuracy))
print("高斯核模型的训练精度: {:.2f}%".format(rbf_accuracy))
print("两者共有的支持向量数量:", len(common_sv))
print("仅在线性核中出现的支持向量索引:", unique_to_linear)
print("仅在高斯核中出现的支持向量索引:", unique_to_rbf)
