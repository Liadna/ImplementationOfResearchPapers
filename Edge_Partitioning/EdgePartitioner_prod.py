import re
import sys
import random
import os
import time
from itertools import islice


# Global Variables
file_limit = 10000
# Neighbors adjacency matrix - node and his list of neighbors
N = {}
# Nodes list
V = []
# Edges list
E = []

# Hyper parameters
alpha = 0
p = 0
m = 0
gamma = 0


def read_graph_data(edges_file):
    """
    Read the graph file specified in the given path and load the graph.
    The file is of the form: <from_node> <to_node> in each line, representing single edge.
    :param edges_file: Path to the graph file
    :return: Tuple of <Vertices, Edges> of the graph.
    """
    print "##################################################"
    print "Reading the graph details of file " + edges_file.split('\\')[-1]
    n = 0
    vertexes = []
    edges = []
    with open(edges_file, 'r') as fid:
        edges_num = 0
        for line in list(islice(fid, file_limit)):
            n += 1
            edges_num += 1
            nodes = re.split(r'[ ,|;\t"]+', str(line).replace('\n', ''))
            # nodes = str(line).replace('\n','').split(' ')
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


def Expand_Min(E, gamma):
    """
    Partition the edges in the edges set through selecting a core vertex and expanding it.
    The expansion continues until there are no neighbors left from the core vertex.
    Uses the original NE heuristic (choosing the vertex with minimum number of neighbors).
    :param E: The edges set
    :param gamma: Stopping criteria: alpha * m / p
    :return: The edges in the k'th partition
    """
    print "Start The Edge Partitioning Algorithm"
    # Initilaizion Phase
    C = []
    S = []
    Ek = []
    count_expands = 0
    while ((len(Ek) <= gamma) and (gamma > 0)):
        # if S and C are the same - pick random node from the Graph that isn't in C
        if set_differnce(S, C):
            x = random.choice(V)
            while (x in C):
                x = random.choice(V)
        # else, Pick the node from S with the minimun number of neighbors out of S
        else:
            min_node = [sys.maxint, None]
            for node in S:
                if not node in C:
                    count = 0
                    for neighbor in N[node]:
                        if not neighbor in S:
                            count += 1
                    if min_node[0] > count:
                        min_node[0] = count
                        min_node[1] = node
            if min_node[1] == None:
                # print "\n!!!You've gone to a dead end. No more neighbors from this vertex!!!"
                break
            x = min_node[1]
        count_expands += 1
        AllocEdges(C, S, Ek, x)
    print "The Edge Partitioning Algorithm - Successfully Finished\n"
    print "Results Report:"
    print "Gamma: {0}".format(gamma)
    # print "C: {0}".format(C)
    print "#Vertexes in C: {0}".format(len(C))
    # print "Ek: {0}".format(Ek)
    print "#Edges in Ek: {0}".format(len(Ek))
    # print "S: {0}".format(S)
    print "#Vertexes in S: {0}".format(len(S))
    print "Number of Expands: {0}".format(count_expands)
    return Ek


def Expand_Max(E, gamma):
    """
    Partition the edges in the edges set through selecting a core vertex and expanding it.
    The expansion continues until there are no neighbors left from the core vertex.
    Uses the original NE heuristic (choosing the vertex with minimum number of neighbors).
    :param E: The edges set
    :param gamma: Stopping criteria: alpha * m / p
    :return: The edges in the k'th partition
    """
    print "Start The Edge Partitioning Algorithm"
    # Initilaizion Phase
    C = []
    S = []
    Ek = []
    count_expands = 0
    while ((len(Ek)<=gamma) and (gamma>0)):
        # if S and C are the same - pick random node from the Graph that isn't in C
        if set_differnce(S, C):
            x = random.choice(V)
            while (x in C):
                x = random.choice(V)
        # else, Pick the node from S with the minimun number of neighbors out of S
        else:
            max_node = [sys.maxint * -1, None]
            for node in S:
                if not node in C:
                    count = 0
                    for neighbor in N[node]:
                        if not neighbor in S:
                            count += 1
                    if max_node[0] < count:
                        max_node[0] = count
                        max_node[1] = node
            if max_node[1] == None:
                # print "\n!!!You've gone to a dead end. No more neighbors from this vertex!!!"
                break
            x = max_node[1]
        count_expands += 1
        AllocEdges(C, S, Ek, x)
    print "The Edge Partitioning Algorithm - Successfully Finished\n"
    print "Results Report:"
    print "Gamma: {0}".format(gamma)
    # print "C: {0}".format(C)
    print "#Vertexes in C: {0}".format(len(C))
    # print "Ek: {0}".format(Ek)
    print "#Edges in Ek: {0}".format(len(Ek))
    # print "S: {0}".format(S)
    print "#Vertexes in S: {0}".format(len(S))
    print "Number of Expands: {0}".format(count_expands)
    return Ek


def set_differnce(A, B):
    """
    Check whether two edges sets are the same (length and the same values)
    :param A: First edge set
    :param B: Second edge set
    :return: True if length values are the same, otherwise False
    """
    if len(A) == len(B):
        for i in A:
            if not i in B:
                return False
        return True
    else:
        return False


def AllocEdges(C, S, Ek, x):
    """
    Allocate new vertices and edges to the k'th partition, moving un-allocated edges from the boundary set to the core set.
    :param C: Core set (allocated edges)
    :param S: Boundary set (candidate edges)
    :param Ek: Current edge set of k'th partition
    :param x: Core vertex to expand and allocate its neighbors
    :return: None
    """
    C.append(x)
    S.append(x)
    for y in N[x]:
        if not y in S:
            S.append(y)
            for z in N[y]:
                if z in S:
                    if (not [y,z] in Ek):
                        Ek.append([y, z])
                    if [y, z] in E:
                        E.remove([y, z])
                    if len(Ek) > gamma:
                        return


def format_time(time):
    """
    Convert the time to hours, minutes and seconds units.
    :param time: Time to be converted
    :return: String of time
    """
    if time < 1:
        s = time * 1000
        h = m = 0
        scale = 'miliseconds'
        return "%02f %s" % (s, scale)
    else:
        m, s = divmod(time, 60)
        h, m = divmod(m, 60)
        scale = 'minutes' if (h == 0 and m > 0) else 'hours' if h > 0 else 'seconds'
        return "%02d:%02d:%02d %s" % (h, m, s, scale)


def run():
    global V
    global gamma
    global m
    global E
    RF = 0  # Replication Factor

    if len(sys.argv) == 4:
        if os.path.exists(sys.argv[1]):
            alpha = float(sys.argv[2])
            p = int(sys.argv[3])
            if alpha <= 0 or p <= 0:
                print("Please enter valid positive alpha and p parameters")
            else:
                V, E = read_graph_data(sys.argv[1])
                m = len(E)
                gamma = alpha * m / p

                start_time = time.time()
                for round in range(p):
                    print("\n@@@@@@@@@ Started round #: %.d @@@@@@@@@" % round)
                    Ek = Expand_Min(E, gamma)
                    V_Ek = set([e[0] for e in Ek])
                    RF += len(V_Ek)

                elapsed_time = time.time() - start_time
                replication_factor = float(RF / len(V)) + 0.5
                print("##########################################")
                print("| Summary of Results Min expansion for hyper-parameters: p=%d, alpha=%.1f" % (p, alpha))
                print("| Replication Factor: %.3f" % replication_factor)
                print("| Execution Time: %s" % format_time(elapsed_time))
                print("##########################################")
        else:
            print "The file doesn't exist. Please enter a correct path of an existing file.\n" \
                  "The file should contain two columns of edges without any other comments"
    else:
        print "Please enter the command: python Edge_Partioning.py edgesfile_path alpha p"


if __name__ == "__main__":
    run()
