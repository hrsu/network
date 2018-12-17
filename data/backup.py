import pandas as pd
import math
import numpy as np
from igraph import *
from PIL import Image
import csv  as  csv
import networkx as nx
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

K = 4
#          0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2
# label= [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]
# label = [0,1,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,1,0]
# label =   []
# label = np.array(label)
np.set_printoptions(threshold=1e6)
colors_type = ['y', 'r', 'g', 'b', 'w']


def NMI(A, B):
    # len(A) should be equal to len(B)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # Mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    # Normalized Mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0 * len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount / total + eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0 * len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + eps, 2)
    MIhat = 2.0 * MI / (Hx + Hy)
    return MIhat


def train(V, r, k, e, lab, rela_err=1e-6):
    m, n = np.shape(V)
    W = np.mat(np.random.random((m, r)))
    H = np.mat(np.random.random((n, r)))

    # W = V
    # H = V
    last_err = 100;

    for x in range(k):
        # error
        V_pre = W * H.T
        E = V - V_pre
        # print E
        err = 0.0
        for i in range(m):
            for j in range(n):
                err += E[i, j] * E[i, j]
        #*****print("第%d次迭代后误差为%f" % (x + 1, err))

        if err < e:
            print("误差阈值收敛.....")
            #nx.draw(G)
            #plt.savefig("karate.png")
            #plt.show()
            break
        if (np.abs(last_err - err) < rela_err):
            print("相对误差收敛......")
            #nx.draw(G)
            #plt.savefig("karate.png")
            #plt.show()
            break

        last_err = err

        a = V.T * W + lab * o * H
        b = H * W.T * W + lab * D * H
        # c = V * H.T
        # d = W * H * H.T
        for i_1 in range(n):
            for j_1 in range(r):
                if b[i_1, j_1] != 0:
                    H[i_1, j_1] = H[i_1, j_1] * a[i_1, j_1] / b[i_1, j_1]

        c = V * H
        d = W * H.T * H
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    W[i_2, j_2] = W[i_2, j_2] * c[i_2, j_2] / d[i_2, j_2]
        if x == k-1:
            nx.draw(G)
            plt.savefig("karate.png")
            plt.show()
        # H = np.array(H)
        # # H每一行向量化
        # q = []
        # for i in range(H.shape[0]):
        #     q.append(H[i])
        # # print(q)
        # # print(W * H)
        #
        #nx.draw(G)
        #plt.savefig("karate.png")
        #plt.show()
        # estimator = KMeans(n_clusters=K, max_iter=300, n_init=40)
        # estimator.fit(q)
        # label_pred = estimator.labels_
        # # print(type(label_pred))
        # print(label_pred)
        # label_pred = np.array(label_pred)
        # for i in range(K):
        #     l1 = []
        #     B = []
        #
        #     for j in range(len(label_pred)):
        #         sum = 0
        #         if label_pred[j] == i:
        #             l1.append(j)
        #         # for h in range(len(l1)):
        #         #     for m in range(r):
        #         #         if H[l1[h], m] != 0:
        #         #             sum = sum - H[l1[h], m] * math.log10(H[l1[h], m])
        #         B.append(sum)
        #     print(l1)
        #     # print(B)

    return W, H


"""
        a = W.T * V + lab*H*o
        b = W.T * W * H + lab*H*D
        # c = V * H.T
        # d = W * H * H.T
        for i_1 in range(r):
            for j_1 in range(n):
                if b[i_1, j_1] != 0:
                    H[i_1, j_1] = H[i_1, j_1] * a[i_1, j_1] / b[i_1, j_1]

        c = V * H.T
        d = W * H * H.T
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    W[i_2, j_2] = W[i_2, j_2] * c[i_2, j_2] / d[i_2, j_2]
"""

def gml2csv(gmlfile):
    g = Graph.Read_GML(gmlfile)
    datas = []
    node_num = 0
    # 读取gml文件中的边
    for line in g.get_edgelist():
        das = []
        for it in line:
            if node_num < int(it):
                node_num = int(it)
            das.append(int(it))
        datas.append(das)
    node_num = node_num + 1
    # print(node_num)
    # print(datas)
    data = np.zeros([node_num, node_num], np.int32)
    # 根据图进行设置邻接矩阵
    for ii in range(len(datas)):
        data[datas[ii][0], datas[ii][1]] = 1
        data[datas[ii][1], datas[ii][0]] = 1
    # 生成networkx的图
    links = []
    for ii in range(len(datas)):
        links = links + [tuple(datas[ii])]
    G = nx.Graph(links)
    return data, G


# def draw_net(net,i):
#     graph = nx.from_numpy_matrix(net)
#     nx.write_pajek(graph,i+'karate.net')

if __name__ == '__main__':
    #import os
    #os.path.abspath('.')+'\data'+filename
    filename = 'karate.gml'
    data, G = gml2csv(filename)
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         data[i, j] = data[j, i]
    #*****print(data)
    N1 = np.mat(data)
    N2 = np.dot(N1, N1)
    N3 = np.dot(N2, N1)
    N4 = np.dot(N3, N1)
    N5 = np.dot(N4, N1)
    N6 = np.dot(N5, N1)
    N7 = np.dot(N6, N1)
    N8 = np.dot(N7, N1)
    N9 = np.dot(N8, N1)
    N10 = np.dot(N9, N1)
    #print("***")
    #print(N1)
    #print("***")
    m = 0
    for i in range(len(N1)):
        for j in range(len(N1)):
            if N1[i, j] == 1:
                m = m + 1
    #*****print(m/len(N1))
    labda = float(len(N1) / m)
    # print(labda)
    data = np.array(data)
    net = np.mat(data)
    s = np.zeros(data.shape, np.float32)
    for i in range(len(s)):
        for j in range(len(s)):
            s[i, j] = labda * N1[i, j]+ labda * labda * N2[i, j]+ labda * labda * labda * N3[i, j] \
                      + labda * labda * labda * labda * N4[i, j] \
                      + labda * labda * labda * labda * labda * N5[i, j] \
                      + labda ** 6 * N6[i, j] + labda ** 7 * N7[i, j] + labda ** 8 * N8[i, j] + labda ** 9 * N9[ i, j] + labda ** 10 * N10[i, j]
    #*****print("_________")
    # for i in range(len(s)):
    #     for j in range(len(s)):
    #         if s[i,j] == 0:
    #             print(i,j)
    z = 0
    for i in range(len(s)):
        for j in range(len(s)):
            z = z + s[i,j]
    #*****print(z/(len(N1)*len(N1)))
    #*****print("+++++++++")
    # print(s)
    #*****print(s)
    o = np.zeros(data.shape, np.float32)
    # o[22, 25] = 1
    # o[39, 38] = 1
    # o[41, 48] = 1
    # o[1, 60] = 1
    # o[31, 32] = 1
    # o[0, 8] = 1
    # o[61,0] = 1
    # o[2, 0] = 1
    # o[26, 57] = 1
    # o[13,60] = 1
    # o[57,60] = 1
    # o[19, 22] = 1
    # o[41, 60] = 1
    # o[2,39] = 1
    # o[1, 5] = 1
    # o[22,26] = 1
    p=0
    for i in range(len(s)):
        for j in range(len(s)):
            if N1[i, j] == 1:
                n = 0
                for k in range(len(s)):
                    if s[i, j] >= s[i, k] and s[i, j] >= s[k, j]:
                        n = n + 1
                    if n >= len(s):
                        o[i, j] = 1
                        p=p+1
                        #*****print(i,j)
    #*****print(p)
    # o = np.mat(o)
    # print(o)
    D = np.zeros(data.shape, np.float32)
    for i in range(len(o)):
        sum = 0
        for j in range(len(o)):
            sum = sum + o[i, j]
        D[i, i] = sum
    D = np.mat(D)
    # print(D)
    # for i in range(len(o)):
    #     for j in range(len(o)):
    #         if o[i,j] > 0:
    #             m = m + 1
    # # print(m)
    # print(o)
    V = N1
    W, H = train(V, K, 20000, 1e-5, 1)

    H = np.array(H)
    # H每一行向量化
    q = []
    for i in range(H.shape[0]):
        q.append(H[i])
    # print(q)
    # print(W * H)

    # nx.draw(G)
    # plt.savefig("karate.png")
    # plt.show()
    estimator = KMeans(n_clusters=K,max_iter=1000, n_init=40)
    estimator.fit(q)
    label_pred = estimator.labels_
    # print(type(label_pred))
    #*****print(label_pred)
    label_pred = np.array(label_pred)

    Division = []
    for i in range(K):
        l1 = []
        for j in range(len(label_pred)):
            if label_pred[j] == i:
                l1.append(j)
        Division.append(l1)
        #print(l1)
    #*****print(Division)
    #return Division

    # print(NMI(label_pred,label))
    # l2 = []
    # for i in range(len(label_pred)):
    #     if label_pred[i] == 1:
    #         l2.append(i)
    # print(l2)

    #
    # x= 0
    # for i in range(len(label)):
    #         if label_pred[i] == label[i]:
    #             x = x + 1.0
    # rate = x/len(label)
    # print('%.2f%%' % (rate * 100))
    # np.savetxt("./label_pred.txt",label_pred)
    # node_color=[]
    # for ii in range(len(label_pred)):
    #     node_color=node_color+[colors_type[label_pred[ii]]]
    # # print(node_color)
    # nx.draw_networkx(G,node_color=node_color)
    # # nx.draw_networkx(G, pos=nx.random_layout(G), node_color=node_color, node_size=10, node_shape='o',width=0.3, style='solid', font_size=8)
    #
    # plt.savefig("new.png")
    # plt.show()


