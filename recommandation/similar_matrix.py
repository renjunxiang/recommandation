import pandas as pd
import numpy as np


class rsm():  # recommendation similar matrix
    # This model is depended on users' choices matrix which is 0-1 matrix
    # If user i buy product j,dataframe.loc['user_i','product_j']=1,else=0
    # Then we calucate similar score:count(A and B)/sqrt(count(A)*count(B))
    # Finally we calucate recommend matrix:similar matrix * users' choices matrix

    def __init__(self, data_raw):
        self.data_raw = data_raw

    def similar_matrix_cal(self):  # calulate similar matrix
        data_raw = self.data_raw
        users_num = data_raw.shape[0]  # user_id is in first column
        data_raw = data_raw.iloc[:, 1:]  # remove first column:user_id
        products = data_raw.columns
        products_num = data_raw.shape[1]
        similar_matrix = pd.DataFrame(
            np.zeros([products_num, products_num]))  # creat products_num*products_num zero-matrix
        for i in range(products_num):
            for j in range(products_num):
                if j != i:
                    # calucate similar point between diffrent products
                    sum_buy = data_raw.agg(sum, axis=0)
                    average_buy = np.sqrt(sum_buy[i] * sum_buy[j])  # weighted denominator
                    for one_user in range(users_num):
                        if data_raw.iloc[one_user, j] == 1 & data_raw.iloc[one_user, i] == 1:  # 客户同时购买i,j
                            similar_matrix.loc[i, j] += 1 / average_buy
        similar_matrix.columns, similar_matrix.index = products, products
        self.similar_matrix = similar_matrix

    def recommend_matrix_cal(self):
        data_raw = self.data_raw
        users = data_raw.iloc[:, 0]
        data_raw = data_raw.iloc[:, 1:]  # remove first column:user_id
        products = data_raw.columns
        recommend_matrix = pd.DataFrame(np.dot(data_raw, self.similar_matrix))
        recommend_matrix.columns, recommend_matrix.index = products, users
        self.recommend_matrix = recommend_matrix

    def recommend_for_user_cal(self, user_id, top=1):
        data_raw = self.data_raw
        products = data_raw.columns
        users = data_raw.iloc[:, 0]
        data_raw.index = users
        data_raw = data_raw.iloc[:, 1:]  # remove first column:user_id
        recommend_matrix = self.recommend_matrix
        user_data = recommend_matrix.loc[user_id, :]
        recommend_for_user = user_data[data_raw.loc[user_id, :] == 0].sort_values(ascending=False)[0:top]
        self.recommend_for_user=recommend_for_user


if __name__ == '__main__':
    # users' choices matrix
    data_raw = pd.DataFrame([['user1', 1, 1, 0, 0],
                             ['user2', 1, 0, 1, 0],
                             ['user3', 0, 0, 1, 1],
                             ['user4', 0, 1, 0, 1],
                             ['user5', 1, 1, 1, 0],
                             ['user6', 0, 0, 0, 1]])
    data_raw.columns = ['user_id', 'products1', 'products2', 'products3', 'products4']
    print('raw data:\n', data_raw)

    # creat model
    model = rsm(data_raw=data_raw)
    model.similar_matrix_cal()
    print('similar_matrix:\n', model.similar_matrix)
    # similar matrix
    # [[0, 0.67, 0.67, 0],
    #  [0.67, 0, 0.33, 0.33],
    #  [0.67, 0.33, 0, 0.33],
    #  [0, 0.33, 0.33, 0]]

    model.recommend_matrix_cal()
    print('recommend_matrix:\n', model.recommend_matrix)

    model.recommend_for_user_cal(user_id='user2', top=1)
    print('recommend_for_user\n', model.recommend_for_user)
