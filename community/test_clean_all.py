import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from models import propagation_with_dynamic_blockage as pro

#先调用clean内的kmeans算法划分社区
#再分别使用每个社区调用动态规划算法


#调用动态规划算法
def getpro(Submatrix):
    G = nx.Graph()
    #edges = np.array([[0, 1, 1, 1, 1, 1, 0, 0],[0, 0, 1, 0, 1, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 1, 0, 0, 0, 1, 1],[0, 0, 0, 0, 0, 1, 0, 1],[0, 0, 0, 0, 0, 1, 1, 0]])
    for i in range(len(Submatrix)):
        for j in range(len(Submatrix)):
            G.add_edge(i, j)

    # nx.draw(G)
    # plt.show()

    lens = len(Submatrix)

    covers = [0]*lens  # 初始节点的评级情况
    times = lens  # 阻塞结束时刻，必须小于等于lambdal
    nodes = [0]*(lens-1) # 节点的感染情况
    nodes.append(1)
    n = lens  # 总的节点个数
    cover = 3  # 一次感染的评级情况
    block_start = 0  # 开始阻塞的时刻
    block_amount = lens//2-3  # 贪心算法每次阻塞的节点个数，每次阻塞后阻塞的个数除以2，如果最后为0则一直为1
    block_duration =lens//2+3   #阻塞时期，处于此时期的节点被阻塞后就不会被感染，超过此时期后，被阻塞也可能会被感染

    pro.propagation_with_dynamic_blockage(covers, times, Submatrix, nodes, n, cover, block_start,
                                                   block_amount, block_duration,'改进后.csv')


#调用clean内的kmeans算法划分社区
from data.clean import *
Division, G1, data1 = Divs(5)
print(Division)
#print(G1._adj)
#print(data1)

datas = np.array(data1)

for i in range(len(Division)):
    Submatrix = datas[Division[i]]  # 先取出想要的行数据
    Submatrix = Submatrix[:, Division[i]]  # 再取出要求的列数据
    print('\n\n{} people\'s result'.format(len(Submatrix)))
    getpro(Submatrix=Submatrix)

'''
for Sub in Division:
    Submatrix = datas[Sub]  # 先取出想要的行数据
    Submatrix = Submatrix[:, Sub]  # 再取出要求的列数据
    print(len(Submatrix))
    #getpro(Submatrix)
'''

'''
edges = np.array(
[
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0]
])
'''
