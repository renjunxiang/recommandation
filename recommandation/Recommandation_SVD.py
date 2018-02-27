import pandas as pd
import numpy as np


class Recommandation_SVD():
    def __init__(self, data, users, products):
        self.data = data
        self.users = users
        self.products = products

    def SVD(self, k=2):
        data = self.data
        # SVD分解，用户矩阵、特征值、产品矩阵的转置
        user_matrix, s, product_matrix_T = np.linalg.svd(data, full_matrices=True)
        user_matrix = np.matrix(user_matrix)
        product_matrix_T = np.matrix(product_matrix_T)
        self.user_matrix = user_matrix  # 用户矩阵
        self.s = s  # 特诊值
        self.product_matrix = product_matrix_T.T  # 产品矩阵

        # 降至k维
        user_matrix_k = user_matrix.T[0:k].T
        s_k = np.diag(s[0:k])
        product_matrix_k = product_matrix_T[0:k].T
        data_predict = user_matrix_k * s_k * (product_matrix_k.T)
        self.k = k  # 降维数
        self.user_matrix_k = user_matrix_k  # 降维后的用户矩阵
        self.s_k = s_k  # 降维后的特诊值矩阵
        self.product_matrix_k = product_matrix_k  # 降维后的产品矩阵
        self.data_predict = data_predict  # SVD分解后用降维的数据还原的原始数据

        var_rate = s[0:k].sum() / s.sum()
        self.var_rate = var_rate  # 特征值中占比
        self.method = 'SVD'

    def user_position(self, x):
        '''
        计算新用户在用户矩阵中的空间位置
        x=np.array([5,5,0,0,0,5])
        :param x: 
        :return: 
        '''
        s_k = self.s_k
        product_matrix_k = self.product_matrix_k

        # 计算用户的空间位置
        x = np.matrix(x)
        position = np.array(x * product_matrix_k * np.linalg.inv(s_k))[0]
        self.user_position = position  # 用户的空间位置
        self.x = np.array(x)[0]

    def user_similarity(self, top=None, method='cosine'):
        '''
        计算新用户和已知用户的相似度
        1.余弦相似度,由于用户评分尺度不同，考察的是相对值，即打分的相对高低
        2.欧氏距离，考察的是绝对值，即打分的绝对高低
        x=np.array([5,5,0,0,0,5])
        :param method:'cosine','Euclidean'
        :param top:取靠前的位次 
        :return: 
        '''
        user_matrix_k = self.user_matrix_k
        user_position = self.user_position

        user_similar_list = []
        if method == 'cosine':
            # 计算和每个已知用户的余弦相似度
            for one_user in user_matrix_k:
                one_user = np.array(one_user)[0]
                cos = np.dot(user_position, one_user) / np.sqrt(
                    np.dot(user_position, user_position) * np.dot(one_user, one_user))
                user_similar_list.append(cos)

        else:
            # 计算和每个已知用户的欧氏距离
            for one_user in user_matrix_k:
                distance = (user_position - one_user) * (user_position - one_user).T
                user_similar_list.append(distance[0, 0])

        # 和每个用户的相似度数据框
        user_similar = pd.DataFrame({'user': self.users, 'similar': user_similar_list},
                                    columns=['user', 'similar'])
        self.user_similar = user_similar  # 相似用户

        # 拼上每个用户的打分
        data_full = pd.DataFrame(self.data, columns=self.products)
        data_full['user'] = self.users
        user_similar_full = pd.merge(left=user_similar, right=data_full, how='left')
        self.user_similar_full = user_similar_full  # 相似用户的评分

        # 给出排名靠前的相似用户
        if top is not None:
            user_similar_top = user_similar.sort_values('similar', ascending=False).iloc[0:top, :]
            self.user_similar_top = user_similar_top
            user_similar_full_top = user_similar_full.sort_values('similar', ascending=False).iloc[0:top, :]
            self.user_similar_full_top = user_similar_full_top

    def recommend_by_product(self, top=3):
        '''
        通过物品坐标和新用户坐标，计算最近的产品
        :param top: 
        :return: 
        '''
        product_matrix_k = self.product_matrix_k
        user_position = np.matrix(self.user_position)
        distances = []
        for i in product_matrix_k:
            distance = (i - user_position) * (i - user_position).T
            distances.append(distance[0, 0])
        product_distances = pd.DataFrame({'products': self.products, 'distances': distances},
                                         columns=['products', 'distances'])
        product_distances.index=self.products
        self.product_distances = product_distances
        if top is not None:
            x = self.x
            recommend_index=list(np.array(self.products)[x==0])
            product_distances_top = product_distances.loc[recommend_index, :].sort_values('distances', ascending=False).iloc[0:top, :]
            self.product_distances_top = product_distances_top
            self.product_top = list(product_distances_top['products'])

    def recommend_by_users(self, top=3):
        '''
        通过已知用户的加权求和，计算最近的产品
        :param top: 
        :return: 
        '''
        user_similar_full = self.user_similar_full
        users_score = (user_similar_full.iloc[:, 2:].T * user_similar_full.iloc[:, 1]).T
        users_score = users_score.agg(sum, axis=0)
        recommend_score_full = pd.DataFrame({'products': self.products,
                                             'score': users_score},
                                            columns=['products', 'score'])
        recommend_score_full.index=self.products
        if top is not None:
            x = self.x
            self.recommend_score_full = recommend_score_full
            recommend_index=list(np.array(self.products)[x==0])
            self.recommend_score_top = recommend_score_full.loc[recommend_index, :].sort_values('score', ascending=False).iloc[0:top, :]
            self.recommend_top = list(self.recommend_score_top['products'])


if __name__ == '__main__':
    data = [
        [5, 5, 3, 0, 5, 5],
        [5, 0, 4, 0, 4, 4],
        [0, 3, 0, 5, 4, 5],
        [5, 4, 3, 3, 5, 5]
    ]
    users = ['user1', 'user2', 'user3', 'user4']
    products = ['product1', 'product2', 'product3', 'product4', 'product5', 'product6']

    model = Recommandation_SVD(data=data,
                               users=users,
                               products=products)
    model.SVD(k=2)
    x = np.array([5, 5, 0, 0, 0, 5])

    # 计算位置坐标
    model.user_position(x=x)
    model.user_similarity(top=2, method='cosine')  # 'Euclidean'
    # print(model.user_similar_top)
    print('相似用户top2的清单：\n', model.user_similar_full_top)

    model.recommend_by_product(top=3)
    print('依据物品矩阵推荐的top3矩阵：\n', model.product_distances_top)
    print('依据物品矩阵推荐的top3：\n', model.product_top)

    model.recommend_by_users(top=3)
    print('依据用户偏好推荐的top3矩阵：\n', model.recommend_score_top)
    print('依据用户偏好推荐的top3：\n', model.recommend_top)
