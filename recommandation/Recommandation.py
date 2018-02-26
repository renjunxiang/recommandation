import pandas as pd
import numpy as np


class Recommandation():
    def __init__(self, data, users, products):
        self.data = data
        self.users = users
        self.products = products

    def SVD(self, k=2):
        data = self.data
        user_matrix, s, product_matrix_T = np.linalg.svd(data, full_matrices=True)
        user_matrix = np.matrix(user_matrix)
        product_matrix_T = np.matrix(product_matrix_T)
        self.user_matrix = user_matrix
        self.s = s
        self.product_matrix = product_matrix_T.T

        user_matrix_k = user_matrix.T[0:k].T
        s_k = np.diag(s[0:k])
        product_matrix_k = product_matrix_T[0:k].T
        data_predict = user_matrix_k * s_k * (product_matrix_k.T)
        self.k = k
        self.user_matrix_k = user_matrix_k
        self.s_k = s_k
        self.product_matrix_k = product_matrix_k
        self.data_predict = data_predict

        var_rate = s[0:k].sum() / s.sum()
        self.var_rate = var_rate
        self.method = 'SVD'

    def user_similar_cal(self, x, top=None):
        '''
        x=np.array([5,5,0,0,0,5])
        :param x: 
        :return: 
        '''
        user_matrix_k = self.user_matrix_k
        s_k = self.s_k
        product_matrix_k = self.product_matrix_k
        x=np.matrix(x)
        position = np.array(x * product_matrix_k * np.linalg.inv(s_k))[0]
        self.user_position = position
        user_similar_list = []
        for one_user in user_matrix_k:
            one_user = np.array(one_user)[0]
            cos = np.dot(position, one_user) / np.sqrt(np.dot(position, position) * np.dot(one_user, one_user))
            user_similar_list.append(cos)
        user_similar = pd.DataFrame({'user': self.users, 'similar': user_similar_list},
                                    columns=['user', 'similar'])
        self.user_similar = user_similar
        if top is not None:
            user_similar_top = user_similar.sort_values('similar', ascending=False).iloc[0:top, :]
            self.user_similar_top = user_similar_top


if __name__ == '__main__':
    data = [
        [5, 5, 3, 0, 5, 5],
        [5, 0, 4, 0, 4, 4],
        [0, 3, 0, 5, 4, 5],
        [5, 4, 3, 3, 5, 5]
    ]
    users = ['user1', 'user2', 'user3', 'user4']
    products = ['product1', 'product2', 'product3', 'product4', 'product5', 'product6']
    model = Recommandation(data=data,
                           users=users,
                           products=products)
    model.SVD(k=2)
    x = np.array([5, 5, 0, 0, 0, 5])
    model.user_similar_cal(x=x,top=2)
    print(model.user_similar_top)
