import networkx as nx
import numpy as np
from igraph import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import datetime

def get_dataG(filen):
    f = open(filen, 'r')
    links = []
    datas = []
    while 1:
        line = f.readline()
        if not line:
            break
        ts = line[:-1].split(' ')
        t = (int(ts[0]), int(ts[1]))
        data = [int(ts[0]), int(ts[1])]
        links.append(t)
        datas.append(data)
    G = nx.Graph(links)

    #node_num = max(max(links)) + 1
    node_num = 115

    data = np.zeros([node_num, node_num], np.int32)
    # 根据图进行设置邻接矩阵
    for s in range(0, len(datas)):
        #print(s)
        data[datas[s][0], datas[s][1]] = 1
        data[datas[s][1], datas[s][0]] = 1

    return data, G

filen = "football.txt"
data, G = get_dataG(filen)

K = 10
np.set_printoptions(threshold=1e6)
colors_type = ['y', 'r', 'g', 'b', 'w']

#定义全局变量
import os
filename = os.path.abspath('..') + '\\dataset\\' + filen
print(filename)
data, G = get_dataG(filename)
o = np.zeros(data.shape, np.float32)
D = np.zeros(data.shape, np.float32)

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

    last_err = 100

    for x in range(k):
        # error
        V_pre = W * H.T
        E = V - V_pre
        # print E
        err = 0.0
        for i in range(m):
            for j in range(n):
                err += E[i, j] * E[i, j]

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
            plt.savefig("football.png")
            plt.show()

    return W, H


def Divs(K1):
    global data, D, G, K
    K = K1

    data1 = data
    N1 = np.mat(data1)
    N2 = np.dot(N1, N1)
    N3 = np.dot(N2, N1)
    N4 = np.dot(N3, N1)
    N5 = np.dot(N4, N1)
    N6 = np.dot(N5, N1)
    N7 = np.dot(N6, N1)
    N8 = np.dot(N7, N1)
    N9 = np.dot(N8, N1)
    N10 = np.dot(N9, N1)
    m = 0
    for i in range(len(N1)):
        for j in range(len(N1)):
            if N1[i, j] == 1:
                m = m + 1
    labda = float(len(N1) / m)
    data = np.array(data1)
    net = np.mat(data)
    s = np.zeros(data.shape, np.float32)
    for i in range(len(s)):
        for j in range(len(s)):
            s[i, j] = labda * N1[i, j]+ labda * labda * N2[i, j]+ labda * labda * labda * N3[i, j] \
                      + labda * labda * labda * labda * N4[i, j] \
                      + labda * labda * labda * labda * labda * N5[i, j] \
                      + labda ** 6 * N6[i, j] + labda ** 7 * N7[i, j] + labda ** 8 * N8[i, j] + labda ** 9 * N9[ i, j] + labda ** 10 * N10[i, j]
    z = 0
    for i in range(len(s)):
        for j in range(len(s)):
            z = z + s[i,j]

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

    for i in range(len(o)):
        sum = 0
        for j in range(len(o)):
            sum = sum + o[i, j]
        D[i, i] = sum
    D = np.mat(D)
    V = N1
    W, H = train(V, K, 20000, 1e-5, 1)

    H = np.array(H)
    # H每一行向量化
    q = []
    for i in range(H.shape[0]):
        q.append(H[i])

    # nx.draw(G)
    # plt.savefig("karate.png")
    # plt.show()
    estimator = KMeans(n_clusters=K,max_iter=1000, n_init=40)
    estimator.fit(q)
    label_pred = estimator.labels_
    label_pred = np.array(label_pred)

    Division = []
    for i in range(K):
        l1 = []
        for j in range(len(label_pred)):
            if label_pred[j] == i:
                l1.append(j)
        Division.append(l1)
    G1 = G
    return Division, G1, data

def main():
    t1 = datetime.datetime.now()
    Division, G1, data = Divs(10)
    #f = open('Division.txt', 'w')
    #f.write(Division)
    #f.close()
    print(Division)
    print("Time usage: ", datetime.datetime.now() - t1)

main()
