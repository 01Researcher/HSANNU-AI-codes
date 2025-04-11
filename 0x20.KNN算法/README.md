# K近邻（K-Nearest Neighbors, KNN）模型学案

---

## 目录
1. **概念讲解**
2. **工具介绍**
3. **实战项目**
4. **应用场景**
5. **总结**

---

## 一、概念讲解

### 1.1 什么是K近邻？
K近邻（K-Nearest Neighbors, KNN）是一种基于实例的**监督学习算法**，用于分类和回归任务。其核心思想是：**相似的数据点在特征空间中彼此靠近**。通过计算新样本与训练数据集中最近邻的K个样本的距离，根据多数表决（分类）或平均值（回归）预测结果。

### 1.2 算法步骤
1. **选择K值**：确定最近邻的数量（K）。
2. **计算距离**：使用欧氏距离、曼哈顿距离等度量方式。
3. **寻找K个最近邻**：找出距离新样本最近的K个训练样本。
4. **决策**：
   - **分类**：统计K个邻居中类别最多的类别。
   - **回归**：取K个邻居目标值的平均值。

### 1.3 关键参数与超参数
- **K值**：较小的K容易过拟合，较大的K可能欠拟合。
- **距离度量**：常用欧氏距离（`metric='euclidean'`）或曼哈顿距离（`metric='manhattan'`）。
- **权重**：是否根据距离加权投票（`weights='distance'`）。

### 1.4 优缺点
- **优点**：简单直观、无需训练过程、适用于多分类。
- **缺点**：计算复杂度高（需存储全部数据）、对噪声敏感、需要特征标准化。

---

## 二、工具介绍

### 2.1 Python库
- **Scikit-learn**：提供`KNeighborsClassifier`和`KNeighborsRegressor`。
- **NumPy & Pandas**：数据处理。
- **Matplotlib & Seaborn**：数据可视化。

### 2.2 安装命令
```bash
pip install scikit-learn numpy pandas matplotlib seaborn
```

---

## 三、实战项目：鸢尾花分类

### 3.1 数据集介绍
- **目标**：根据花萼和花瓣的尺寸预测鸢尾花种类（Setosa, Versicolor, Virginica）。
- **特征**：`sepal_length`, `sepal_width`, `petal_length`, `petal_width`。

### 3.2 代码实现

# 实战项目：手写KNN算法与Sklearn实战

---

## 项目目标
1. **阶段一**：不使用机器学习库，手动实现KNN分类算法。
2. **阶段二**：使用Scikit-learn库完成相同任务，对比实现效率。

---

## 任务说明
**数据集**：鸢尾花分类数据集（150条数据，4个特征，3个类别）  
**目标**：根据花萼和花瓣的尺寸预测鸢尾花种类。

---

## 阶段一：手动实现KNN算法

### 1. 需要使用的库和函数
- **NumPy**：用于数值计算（如距离计算、数组操作）
  - 关键函数：`np.sqrt()`, `np.sum()`, `np.argsort()`
- **Pandas**：加载和查看数据
  - 关键函数：`pd.read_csv()`, `df.head()`
- **Matplotlib**（可选）：数据可视化

### 2. 任务步骤
#### 步骤1：加载数据
```python
import numpy as np
import pandas as pd

# 手动下载数据集（或从sklearn.datasets获取）
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
data = pd.read_csv(url, names=columns)
```

#### 步骤2：数据预处理
- 将类别标签转换为数字（例如Setosa=0, Versicolor=1, Virginica=2）
- 将数据划分为训练集（80%）和测试集（20%）
- **标准化特征值**（提示：使用`(x - mean)/std`）

#### 步骤3：实现KNN核心函数
```python
def euclidean_distance(x1, x2):
    """计算两个样本之间的欧氏距离"""
    return np.sqrt(np.sum((x1 - x2)**2))

def predict_knn(X_train, y_train, x_test, k=3):
    """
    手动实现KNN预测
    参数说明：
        X_train: 训练集特征（n_samples x n_features）
        y_train: 训练集标签
        x_test: 单个测试样本
        k: 最近邻数量
    """
    # 1. 计算x_test与所有训练样本的距离
    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    
    # 2. 找到距离最近的k个样本的索引
    k_indices = np.argsort(distances)[:k]
    
    # 3. 统计k个邻居的类别
    k_labels = [y_train[i] for i in k_indices]
    
    # 4. 多数投票决定预测结果
    return max(set(k_labels), key=k_labels.count)
```

#### 步骤4：评估模型
- 对测试集所有样本调用`predict_knn`函数
- 计算准确率（正确预测数 / 总样本数）

---

## 阶段二：使用Scikit-learn实现

### 1. 需要使用的库和函数
- **Scikit-learn**：
  - `KNeighborsClassifier`：KNN分类器
  - `train_test_split`：划分数据集
  - `StandardScaler`：数据标准化
  - `accuracy_score`：计算准确率

### 2. 任务步骤
#### 步骤1：使用Sklearn加载数据
```python
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
```

#### 步骤2：数据预处理
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 标准化（必须与阶段一保持一致）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

#### 步骤3：训练与预测
```python
from sklearn.neighbors import KNeighborsClassifier

# 创建KNN模型（尝试k=3,5,7）
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

#### 步骤4：模型优化
- 使用交叉验证选择最佳k值（提示：`GridSearchCV`）
- 比较不同距离度量（如曼哈顿距离）的效果

---

## 思考题
1. 手动实现的KNN和Sklearn版本在运行速度上有何差异？为什么？
2. 如果不做数据标准化，会对KNN产生什么影响？
3. 当特征数量增加到100维时，欧氏距离是否仍然有效？（了解"维度灾难"概念）

## 四、应用场景
- **分类**：垃圾邮件检测、图像识别。
- **回归**：房价预测、用户评分预测。
- **推荐系统**：基于用户/物品相似性的推荐。

---

## 五、总结
- **KNN是懒惰学习算法**：无需显式训练，但预测时计算量大。
- **适合小数据集**：大数据集建议使用近似最近邻算法（如Annoy、Faiss）。
- **标准化是关键**：确保所有特征具有相似的尺度。

---

## 扩展与练习
1. 尝试在KNN中使用不同的距离度量（如曼哈顿距离）。
2. 用KNN回归预测波士顿房价数据集。
3. 阅读论文《A Few Useful Things to Know About Machine Learning》了解KNN的局限性。
