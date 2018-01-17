# categorization

## 语言
Python3.5<br>
## 依赖库
pandas=0.21.0<br>
numpy=1.13.1<br>
scikit-learn=0.19.1<br>
## 原理说明
将分词后的样本（英文自动按空格，中文分词后需按空格分割）通过sklearn.feature_extraction.text模块转为哈希格式减小存储开销，然后通过常用的机器学习分类模型如SVM和KNN进行学习和预测。本质为将文本转为稀疏矩阵作为训练集的数据，结合标签进行监督学习。
![ex1](https://github.com/renjunxiang/machine-learning/blob/master/NLP/文本分类.png)

