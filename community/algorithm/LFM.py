import random
import networkx as nx
import os
import numpy as np
import datetime
import os

class Community():
    ''' use set operation to optimize calculation '''
    
    def __init__(self,G,alpha=1.0):
        self._G = G
        self._alpha = alpha
        self._nodes = set()
        self._k_in = 0
        self._k_out = 0
        
    def add_node(self,node):
        neighbors = set(self._G.neighbors(node))
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        self._nodes.add(node)
        self._k_in += 2*node_k_in
        self._k_out = self._k_out+node_k_out-node_k_in
        
    def remove_vertex(self,node):
        neighbors = set(self._G.neighbors(node))
        community_nodes = self._nodes
        node_k_in = len(neighbors & community_nodes)
        node_k_out = len(neighbors) - node_k_in
        self._nodes.remove(node)
        self._k_in -= 2*node_k_in
        self._k_out = self._k_out - node_k_out+node_k_in
        
    def cal_add_fitness(self,node):
        neighbors = set(self._G.neighbors(node))
        old_k_in = self._k_in
        old_k_out = self._k_out
        vertex_k_in = len(neighbors & self._nodes)
        vertex_k_out = len(neighbors) - vertex_k_in 
        new_k_in = old_k_in + 2*vertex_k_in
        new_k_out = old_k_out + vertex_k_out-vertex_k_in
        new_fitness = new_k_in/(new_k_in+new_k_out)**self._alpha
        old_fitness = old_k_in/(old_k_in+old_k_out)**self._alpha
        return new_fitness-old_fitness
    
    def cal_remove_fitness(self,node):
        neighbors = set(self._G.neighbors(node))
        new_k_in = self._k_in
        new_k_out = self._k_out
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        old_k_in = new_k_in - 2*node_k_in
        old_k_out = new_k_out - node_k_out + node_k_in
        old_fitness = old_k_in/(old_k_in+old_k_out)**self._alpha 
        new_fitness = new_k_in/(new_k_in+new_k_out)**self._alpha
        return new_fitness-old_fitness
    
    def recalculate(self):
        for vid in self._nodes:
            fitness = self.cal_remove_fitness(vid)
            if fitness < 0.0:
                return vid
        return None
    
    def get_neighbors(self):
        neighbors = set()
        for node in self._nodes:
            neighbors.update(set(self._G.neighbors(node)) - self._nodes)
        return neighbors
    
    def get_fitness(self):
        return float(self._k_in)/((self._k_in+self._k_out) ** self._alpha)

class LFM():
    
    def __init__(self, G, alpha):
        self._G = G
        self._alpha = alpha
        
    def execute(self):
        communities = []
        node_not_include = list(self._G.node.keys())[:]
        while(len(node_not_include) != 0):
            c = Community(self._G, self._alpha)
            # randomly select a seed node
            seed = random.choice(node_not_include)
            c.add_node(seed)
             
            to_be_examined = c.get_neighbors()
            while(to_be_examined):
                #largest fitness to be added
                m = {}
                for node in to_be_examined:
                    fitness = c.cal_add_fitness(node)
                    m[node] = fitness
                to_be_add = sorted(m.items(),key=lambda x:x[1],reverse = True)[0]
                 
                #stop condition
                if(to_be_add[1] < 0.0):
                    break
                c.add_node(to_be_add[0])
                 
                to_be_remove = c.recalculate()
                #while(to_be_remove != None):
                #    c.remove_node(to_be_remove)
                #    to_be_remove = c.recalculate()
                    
                to_be_examined = c.get_neighbors()
                                     
            for node in c._nodes:
                if(node in node_not_include):
                    node_not_include.remove(node)
            communities.append(c._nodes)
        return communities


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

    # node_num = max(max(links)) + 1
    node_num = 6006
    #email: 1005
    #bitcoin: 6006

    data = np.zeros([node_num, node_num], np.int32)
    # 根据图进行设置邻接矩阵
    for s in range(0, len(datas)):
        # print(s)
        data[datas[s][0], datas[s][1]] = 1
        data[datas[s][1], datas[s][0]] = 1

    return data, G


def main(commu):
    #G = nx.karate_club_graph()
    path = os.path.abspath('.') + '\\data\\'
    t1 = datetime.datetime.now()
    file = path + str(commu) + '.txt'
    data, G = get_dataG(file)
    algorithm = LFM(G, 0.8)
    communities = algorithm.execute()
    new_communities = []
    for c in communities:
        if len(c) > 20:
            new_communities.append(list(c))
            print (list(c))
    print("Total " + str(len(new_communities)) + " communities")
    print("Time usage:", datetime.datetime.now() - t1)
    return new_communities, G, data


#main('club')
