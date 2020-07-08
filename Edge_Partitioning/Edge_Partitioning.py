import sys
import random
import os

#Global Variables
N = {} #Neighbors adjacency matrix - node and his list of neighcors
V = []
E = []
alpha = 1
p = 50
m = 0
gamma = alpha*m/p

def read_graph_data(edges_file):
    print "Reading the graph details"
    vertexes = []
    edges = []
    with open(edges_file,'r') as fid:
        edges_num = 0
        for line in fid:
            edges_num += 1
            nodes = str(line).replace('\n','').split(' ')
            if nodes[0] in N:
                if (not nodes[1] in N[nodes[0]]):
                    N[nodes[0]].append(nodes[1])
            else:
                N[nodes[0]] = [nodes[1]]
            if nodes[1] in N:
                if (not nodes[0] in N[nodes[1]]):
                    N[nodes[1]].append(nodes[0])
            else:
                N[nodes[1]] = [nodes[0]]
            if not(nodes[0]) in vertexes:
                vertexes.append(nodes[0])
            if not(nodes[1]) in vertexes:
                vertexes.append(nodes[1])
            edges.append([nodes[0], nodes[1]])
    print "Finished reading the graph details"
    return vertexes, edges

def Expand(E, gamma):
    print "\nStart The Edge Partitioning Algorithm"
    #Initilaizion Phase
    C = []
    S = []
    Ek = []
    count_expands = 0
    while ((len(Ek)<=gamma) and (gamma>0)):
        #if S and C are the same - pick random node from the Graph that isn't in C
        if set_differnce(S,C):
            x = random.choice(V)
            while (x in C):
                x = random.choice(V)
        #else, Pick the node from S with the minimun number of neighbors out of S
        else:
            max_node = [sys.minint,None]
            for node in S:
                if not node in C:
                    count = 0
                    for neighbor in N[node]:
                        if not neighbor in S:
                            count +=1
                    if max_node[0]<count:
                        max_node[0] = count
                        max_node[1] = node
            if min_node[1] == None:
                print "\n!!!You've gone to a dead end. No more neighbors from this vertex!!!\n"
                break
            x = min_node[1]
        count_expands +=1
        AllocEdges(C, S, Ek, x)
    print "The Edge Partitioning Algorithm - Successfully Finished\n"
    print "Results Report:"
    print "Gamma: {0}".format(gamma)
    print "C: {0}".format(C)
    print "#Vertexes in C: {0}".format(len(C))
    print "Ek: {0}".format(Ek)
    print "#Edges in Ek: {0}".format(len(Ek))
    print "S: {0}".format(S)
    print "#Vertexes in S: {0}".format(len(S))
    print "Number of Expands: {0}".format(count_expands)

# returns true if the groups are the same (length and the same values)
def set_differnce(A,B):
    if len(A) == len(B):
        for i in A:
            if not i in B:
                return False
        return True
    else:
        return False

def AllocEdges(C, S, Ek, x):
    C.append(x)
    S.append(x)
    for y in N[x]:
        if not y in S:
            S.append(y)
            for z in N[y]:
                if z in S:
                    if (not [y,z] in Ek):
                        Ek.append([y,z])
                    if [y,z] in E:
                        E.remove([y,z])
                    if len(Ek)>gamma:
                        return

def run():
    if len(sys.argv)==2:
        if os.path.exists(sys.argv[1]):
            global V
            V, E = read_graph_data(sys.argv[1])
            m = len(E)
            gamma = alpha * m / p
            Expand(E, gamma)
        else:
            print "The file doesn't exist. Please enter a correct path of an existing file."
    else:
        print "Please enter the command: python Edge_Partioning.py edgesfile_path"

if __name__ == "__main__":
    run()