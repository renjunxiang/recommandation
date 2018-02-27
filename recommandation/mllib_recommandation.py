from pyspark import SparkConf, SparkContext
import pandas as pd
import numpy as np

from pyspark.mllib.recommendation import ALS #用户和产品矩阵都可以聚类做分类,也可以根据预测值做推荐

conf = SparkConf().setMaster("local[*]").setAppName("First_App")
sc = SparkContext(conf=conf)
sc.setLogLevel('WARN')

# data = [[['uid', np.random.randint(1, 5)],
#          ['pid', np.random.randint(1, 5)],
#          ['rate', np.random.randint(1, 11)]] for i in range(20)]
data_raw = [[1, 1, 1], [1, 2, 1],
            [2, 1, 1], [2, 3, 1],
            [3, 3, 1], [3, 4, 1],
            [4, 2, 1], [4, 4, 1],
            [5, 1, 1], [5, 2, 1], [5, 3, 1],
            [6, 4, 1]]
pd.DataFrame(data_raw)
data=sc.parallelize(data_raw)
model=ALS.train(ratings=data,rank=5,iterations=10,lambda_=0.01)
model.userFeatures().collect()#用户因子
model.productFeatures().collect()#产品因子
model.recommendProductsForUsers(2).collect()#给出每个客户得分前两位产品
model.recommendUsersForProducts(2).collect()#给出每个产品得分前两位的客户
model.predict(1,2)#给出客户1对产品2的预测得分
recommend_products_for_users_1=model.predictAll(data.map(lambda x:[x[0],x[1]])).collect()#查看每一个样本的预测得分
# model=ALS.trainImplicit()#用于隐式反馈数据


