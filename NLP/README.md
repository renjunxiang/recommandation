# categorization

## 语言
Python3.5<br>
## 依赖库
pandas=0.21.0<br>
numpy=1.13.1<br>
scikit-learn=0.19.1<br>
jieba=0.39<br>
## 脚本说明
### 文本转稀疏矩阵：sentence_transform.py
先用jieba分词，再提供两种稀疏矩阵转换方式：1.转one-hot形式的矩阵，使用pandas的value_counts计数后转dataframe；2.sklearn.feature_extraction.text转成哈希表结构的矩阵。<br>
![ex1](https://github.com/renjunxiang/machine-learning/blob/master/NLP/文本转矩阵.png)
### 监督学习：supervised_classify.py
利用sentence_transform.py文本转稀疏矩阵后，通过sklearn.feature_extraction.text模块转为哈希格式减小存储开销，然后通过常用的机器学习分类模型如SVM和KNN进行学习和预测。本质为将文本转为稀疏矩阵作为训练集的数据，结合标签进行监督学习。<br>
![ex1](https://github.com/renjunxiang/machine-learning/blob/master/NLP/picture/文本分类.png)
### 非监督学习：LDA.py
利用sentence_transform.py文本转稀疏矩阵后，对稀疏矩阵进行ALS分解，转为文本-主题矩阵*主题-词语矩阵。<br>
![ex2](https://github.com/renjunxiang/machine-learning/blob/master/NLP/picture/文本主题分类数据.png)
![ex3](https://github.com/renjunxiang/machine-learning/blob/master/NLP/picture/文本主题分类.png)

