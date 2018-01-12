import pandas as pd
import numpy as np


class ALS():  # recommendation similar matrix
    # R=P*Q.T,step迭代次数,alpha学习率,beta正则系数
    # 1.计算梯度(R里面的每个元素逐个计算)
    # loss=sum(e[i,j]**2)=sum((R[i][j] - np.dot(P[i,:], Q[:,j]))**2)原始误差
    # loss_full=loss+0.5*beta*(P**2+Q**2)带上正则项
    # P[i][k]的偏导=2*e[i,j]*-Q[k,j]+beta*P[i][k]
    # 反向更新P:[i][k]=P[i][k]-alpha*P[i][k]的偏导
    # 计算整体误差,sum(loss_full)

    def __init__(self, R, K=2, alpha=0.0002, beta=0.02,
                 steps=5000, error=0.1):
        self.R = R
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.steps = steps
        self.error = error

    def ALS_matrix(self):  # calulate similar matrix
        R = self.R
        K = self.K
        alpha = self.alpha
        beta = self.beta
        steps = self.steps
        error = self.error

        users = R.iloc[:, 0]
        R = R.iloc[:, 1:]  # remove first column:user_id
        products = R.columns
        # R=P*Q.T,step迭代次数,alpha学习率,beta正则系数
        # 1.计算梯度(R里面的每个元素逐个计算)
        # loss=sum(e[i,j]**2)=sum((R[i][j] - np.dot(P[i,:], Q[:,j]))**2)原始误差
        # loss_full=loss+0.5*beta*(P**2+Q**2)带上正则项
        # P[i][k]的偏导=2*e[i,j]*-Q[k,j]+beta*P[i][k]
        # 反向更新P:[i][k]=P[i][k]-alpha*P[i][k]的偏导
        # 计算整体误差,sum(loss_full)

        M, N = R.shape[0], R.shape[1]
        P = np.random.rand(M, K)  # 先生成随机数矩阵
        Q = np.random.rand(N, K)
        Q = Q.T
        for step in range(steps):  # 迭代5000次
            e = 0  # 先计算误差,小于阈值就停止
            not_zero = 0
            for i in range(M):
                for j in range(N):
                    if R.iloc[i,j] > 0:
                        not_zero += 1
                        e = e + pow(R.iloc[i,j] - np.dot(P[i, :], Q[:, j]), 2)  # 当R[i][j]位置不等于0,计算误差
                        for k in range(K):
                            e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))  # 加上正则项惩罚
            e = e / max(not_zero, 1)
            if e < error:  # loss function < 0.001
                break

            for i in range(M):  # row
                for j in range(N):  # columns
                    if R.iloc[i,j] > 0:
                        eij = R.iloc[i,j] - np.dot(P[i, :], Q[:, j])  # 当R[i][j]位置不等于0,计算梯度并反向更新
                        for k in range(K):
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])  # 带正则项的偏导*alpha
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        P = pd.DataFrame(P)
        Q = pd.DataFrame(Q)
        P.index=users
        Q.columns=products
        return P, Q.T, e

    def recommend_matrix(self):
        R = self.R
        K = self.K
        alpha = self.alpha
        beta = self.beta
        steps = self.steps
        error = self.error

        users = R.iloc[:, 0]
        R = R.iloc[:, 1:]  # remove first column:user_id
        products = R.columns
        P, Q, e = self.ALS_matrix()
        result = pd.DataFrame(np.dot(P, Q.T))
        result.columns, result.index = products, users
        return P, Q, result

    def recommend_for_user(self, user_id, num=1):
        R = self.R

        users = R.iloc[:, 0]
        R = R.iloc[:, 1:]  # remove first column:user_id
        R.index = users
        P, Q, recommend_matrix = self.recommend_matrix()
        user_data = recommend_matrix.loc[user_id, :]
        result = user_data[R.loc[user_id, :] == 0].sort_values(ascending=False)[0:num]
        return result


if __name__ == '__main__':
    # user-products data
    R = np.array([np.random.randint(0, 6, 4) for i in range(5)])
    R = pd.concat([pd.DataFrame({'user_id': ['user' + str(i) for i in range(5)]}),
                   pd.DataFrame(R)], axis=1)
    R.columns = ['user_id']+['products'+str(i) for i in range(4)]


    print('raw data:\n', R)

    # model of recommendation similar matrix
    model = ALS(R, K=2, alpha=0.0002, beta=0.02,
                steps=5000, error=0.1)
    P, Q, result = model.recommend_matrix()
    print('users_matrix:\n', P)
    print('products_matrix:\n', Q)
    print('recommend_matrix:\n', result)
    print('recommend_for_user:\n', model.recommend_for_user(user_id='user2', num=1))
