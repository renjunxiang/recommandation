import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import jieba

jieba.setLogLevel('WARN')

class text_classify():
    def __init__(self,
                 language='English',
                 model_exist=False,
                 model_path=None,  # 模型路径
                 model_name='SVM',  # SVM,KNN,Logistic
                 hashmodel='CountVectorizer',  # 哈希方式:CountVectorizer,TfidfTransformer,HashingVectorizer
                 savemodel=False,
                 train_dataset=None,  # 训练集[[数据],[标签]]
                 test_data=None):  # 测试集[数据]
        self.language=language
        self.model_exist = model_exist
        self.model_path = model_path
        self.model_name = model_name
        self.hashmodel = hashmodel
        self.savemodel = savemodel
        self.train_dataset = train_dataset
        self.test_data = test_data

    def sentence_cut(self): #对中文进行分词处理
        train_dataset = self.train_dataset
        test_data = self.test_data
        if self.language == 'Chinese':
            train_dataset[0]=[' '.join(jieba.lcut(i)) for i in train_dataset[0]]
            test_data=[' '.join(jieba.lcut(i)) for i in test_data]
        return train_dataset,test_data

    def data_transform(self):  # 全部数据转稀疏
        train_dataset,test_data = self.sentence_cut()
        train_data, train_label = train_dataset[0], train_dataset[1]  # 分离数据和标签
        hashmodel = self.hashmodel
        if hashmodel == 'CountVectorizer':  # 只计数
            count_train = CountVectorizer()
            train_data_hashcount = count_train.fit_transform(train_data)  # 训练数据转哈希计数
            count_test = CountVectorizer(vocabulary=count_train.vocabulary_)  # 测试数据调用训练词库
            test_data_hashcount = count_test.fit_transform(test_data)  # 测试数据转哈希计数
        elif hashmodel == 'TfidfTransformer':  # 计数后计算tf-idf
            count_train = CountVectorizer()
            train_data_hashcount = count_train.fit_transform(train_data)  # 训练数据转哈希计数
            count_test = CountVectorizer(vocabulary=count_train.vocabulary_)  # 测试数据调用训练词库
            test_data_hashcount = count_test.fit_transform(test_data)  # 测试数据转哈希计数
            tfidftransformer = TfidfTransformer()
            train_data_hashcount = tfidftransformer.fit(train_data_hashcount).transform(train_data_hashcount)
            test_data_hashcount = tfidftransformer.fit(test_data_hashcount).transform(test_data_hashcount)
        elif hashmodel == 'HashingVectorizer':  # 哈希计算
            vectorizer = HashingVectorizer(stop_words=None, n_features=10000)
            train_data_hashcount = vectorizer.fit_transform(train_data)  # 训练数据转哈希后的特征,避免键值重叠导致过大有一个计算的
            test_data_hashcount = vectorizer.fit_transform(test_data)  # 测试数据转哈希后的特征
        return train_data_hashcount, train_label, test_data_hashcount

    def model_train(self):
        train_data_hashcount, train_label, test_data_hashcount = self.data_transform()
        model_path = self.model_path
        if self.model_exist == False:  # 如果不存在模型,调训练集训练
            model_name = self.model_name
            if model_name == 'KNN':
                model = KNeighborsClassifier(n_neighbors=min(len(train_label), 5))  # 调用KNN,近邻=5
                model.fit(train_data_hashcount, train_label)
            elif model_name == 'SVM':
                model = SVC(kernel='linear', C=1.0)  # 核函数为线性,惩罚系数为1
                model.fit(train_data_hashcount, train_label)
            elif model_name == 'Logistic':
                model = LogisticRegression(solver='liblinear',C=1.0)  # 核函数为线性,惩罚系数为1
                model.fit(train_data_hashcount, train_label)

            if self.savemodel == True:
                joblib.dump(model, model_path)  # 保存模型
        else:  # 存在模型则直接调用
            model = joblib.load(model_path)
        return model

    def model_predict(self):
        model = self.model_train()
        train_data_hashcount, train_label, test_data_hashcount = self.data_transform()
        result = model.predict(test_data_hashcount)#对测试集进行预测
        return result


if __name__ == '__main__':
    print('example:English')
    train_dataset = [['he likes apple',
                      'he really likes apple',
                      'he hates apple',
                      'he really hates apple'],
                     [1, 1, 0, 0]]
    test_data = ['she likes apple',
                 'she really hates apple',
                 'tom likes apple',
                 'tom really hates apple'
                 ]
    test_label = [1, 0, 1, 0]
    text_classify_try = text_classify(train_dataset=train_dataset,
                                      test_data=test_data,
                                      model_name='SVM')
    result = text_classify_try.model_predict()
    print('score:', np.sum(result == np.array(test_label)) / len(result))
    result = pd.DataFrame({'data': test_data,
                           'label': test_label,
                           'predict': result},
                          columns=['data', 'label', 'predict'])
    print('train data\n',
          pd.DataFrame({'data':train_dataset[0],
                        'label':train_dataset[1]},
                       columns=['data','label']))
    print('test\n',result)

    print('example:Chinese')
    train_dataset = [['国王喜欢吃苹果',
                      '国王非常喜欢吃苹果',
                      '国王讨厌吃苹果',
                      '国王非常讨厌吃苹果'],
                     [1, 1, 0, 0]]
    print('train data\n',
          pd.DataFrame({'data':train_dataset[0],
                        'label':train_dataset[1]},
                       columns=['data','label']))
    test_data = ['涛哥喜欢吃苹果',
                 '涛哥讨厌吃苹果',
                 '涛哥非常喜欢吃苹果',
                 '涛哥非常讨厌吃苹果'
                 ]
    test_label = [1, 0, 1, 0]
    text_classify_try = text_classify(train_dataset=train_dataset,
                                      test_data=test_data,
                                      model_name='SVM',
                                      language='Chinese')
    result = text_classify_try.model_predict()
    print('score:', np.sum(result == np.array(test_label)) / len(result))
    result = pd.DataFrame({'data': test_data,
                           'label': test_label,
                           'predict': result},
                          columns=['data', 'label', 'predict'])
    print('test\n',result)
