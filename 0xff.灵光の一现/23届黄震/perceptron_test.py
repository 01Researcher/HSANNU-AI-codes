import numpy as np  # 进行矩阵运算和变换
from sklearn.datasets import load_digits  # 得到手写数字数据集
import matplotlib.pyplot as plt  # 进行acc变化曲线的绘制


def evaluate(w, b, datas, labels):
    y_hat = np.matmul(datas, w) + b
    y_pred = (y_hat >= 0).astype(int)
    y_pred[y_pred == 0] = -1
    acc = np.mean(y_pred == labels)

    return acc


# 绘制 acc 曲线
def display_data(x, y, xlabel, y_label, title=None):
    plt.plot(x, y)
    # 添加网格信息
    plt.grid(True, linestyle='--', alpha=0.5)  # 默认是True，风格设置为虚线，alpha为透明度
    # 添加坐标轴标签
    plt.xlabel(xlabel)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


# 得到数据
digits = load_digits()
# 特征空间
features = digits['data']

# 将数字 0~4 分类类别-1; 数字 5~9 分为类别2
labels = (digits['target'] > 4).astype(int)  # 将大于 4 的标签转换为 1, 小于 4 的转换为 0
labels[labels == 0] = -1  # 将小于 4 的标签转换为 -1

shuffle_indices = np.random.permutation(features.shape[0])
# 得到长度为数据个数并打乱顺序的数组, 来作为索引

features = features[shuffle_indices]  # 通过索引得到打乱后的特征和标签
labels = labels[shuffle_indices]

train_num = int(features.shape[0] * 0.8)  # 按照 4:1 划分数据集
train_datas, train_labels = features[:train_num, :], labels[: train_num]
test_datas, test_labels = features[train_num:, :], labels[train_num:]

# 输出数据形状
print(test_datas.shape, test_labels.shape)  # (360, 64) (360,)
print(train_datas.shape, train_labels.shape)  # (1437, 64) (1437,)

BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 100

w = np.zeros((train_datas.shape[1],))
# w 的 shape 为: (64, ), 一维数组, 只有一个维度, 并且该维度长度为64
b = 0
acc_list = []  # 存放测试集准确率, 方便后面绘图

for epoch in range(EPOCHS):
    cur = 0
    while cur < train_num:  # 训练使用了的数据长度小于训练数据总长度
        # 取出当前的训练数据
        current_data = train_datas[cur: cur + BATCH_SIZE, :]  # shape : (64, 64)
        current_labels = train_labels[cur: cur + BATCH_SIZE]  # shape: (64,)

        y_hat = np.matmul(current_data, w) + b  # np.matmul() 表示进行向量间的点乘

        # 将得到的结果转换为对应标签
        y_hat[y_hat >= 0] = 1
        y_hat[y_hat < 0] = -1

        # 得到分类错误的数据索的引矩阵
        flags = y_hat != current_labels
        # flags 的形状为: (64,)

        w += np.mean(current_labels[flags].reshape((-1, 1)) * current_data[flags], 0)
        b += np.mean(current_labels[flags])

        cur += BATCH_SIZE  # 对下一批次数据进行计算

    acc = evaluate(w, b, test_datas, test_labels)  # 对测试集进行 acc 计算
    print('Epoch:%d, accuracy:%.4f'%(epoch+1, acc))
    acc_list.append(acc)

x = range(1, len(acc_list) + 1)
display_data(x, acc_list, 'epoch', 'acc')
