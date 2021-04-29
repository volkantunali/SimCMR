# -*- coding: utf-8 -*-
import networkx as nx
import tunali_community as tc
import sklearn.metrics as mt
from networkx.algorithms import community
from community import community_louvain
import os
import time
import csv
import infomap
from argparse import Namespace
import  oslom_runner


# ------- Utilities ------------------------
# Some algorithms produce partition sets,
# while some produce community assighments.
# In order to unify their results, these
# two utility functions are helpful.
def partition_to_clustering(G, partition):
    y_true = {}

    temp = {}
    for (i, part) in enumerate(partition, 0):
        for node in part:
            temp[node] = i

    for node in G:
        y_true[node] = temp[node]

    return y_true

def clustering_to_partition(nodeToCluster, k):
    partitionTemp = [set([]) for x in range(k)]
    for node, cluster in nodeToCluster.items():
        if cluster < 0 or cluster >= k:
            print("hatalı nodeToCluster: ", node, "-->", cluster)
        partitionTemp[cluster].add(node)
    return partitionTemp
# ------- Utilities ------------------------
    

def write_a_result(file_object, file_name, random_seed, n, m, num_clusters, original_modularity,
                   algorithm, num_clusters_found, modularity, nmi, ami, ari, elapsed):
        file_object.write(file_name +
                      "\t" + str(random_seed) +
                      "\t" + str(n) +
                      "\t" + str(m) +
                      "\t" + str(num_clusters) +
                      "\t" + str(original_modularity) +
                      "\t" + algorithm +
                      "\t" + str(num_clusters_found) +
                      "\t" + str(modularity) +
                      "\t" + str(nmi) +
                      "\t" + str(ami) +
                      "\t" + str(ari) +
                      "\t" + str(elapsed) +
                      "\n")

# ========================================================
# MAIN program begins here.
# ========================================================
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# Determine the algorithms to run.
# Sometimes, we need to run algorithm one by one, especially on large networks.
do_KM1 = True
do_BKM1 = True
do_GMO = True
do_LPA = True
do_LVN = True
do_SIMCMR = True
do_INFOMAP = True
do_OSLOM = True

# Name of the file to contain the experiment results.
results_file_name = "lfr_test_result_1000.txt"
if os.path.exists(results_file_name):
    os.remove(results_file_name)

# Path of the graph files to be considered
graph_files = []
path = ".\\graphs\\1000"
for r, d, f in os.walk(path):
    for file in f:
        if file.endswith(".network.txt"):
            full_path = os.path.join(r, file)
            print(full_path)
            # print(file)
            graph_files.append(full_path)



for file_name in graph_files:
    print("Processing file:", file_name)

    # Appending to the results file. No overwriting.
    file_object = open(results_file_name, "a")

    G = nx.Graph()
    y_true = {}

    community_dat = file_name.replace(".network.txt", ".community.txt")
    with open(community_dat) as infile:
        csv_reader = csv.reader(infile,  delimiter='\t')
        for row in csv_reader:
            node = int(row[0]) - 1
            comm = int(row[1]) - 1
            y_true[node] = comm
            G.add_node(node)

    with open(file_name) as infile:
        csv_reader = csv.reader(infile,  delimiter='\t')
        for row in csv_reader:
            node1 = int(row[0]) - 1
            node2 = int(row[1]) - 1
            G.add_edge(node1, node2)


    partition = clustering_to_partition(y_true, len(set(y_true.values())))

    n = len(G.nodes)
    m = len(G.edges)

    num_clusters = len(partition)
    original_modularity = community_louvain.modularity(y_true, G)
    y_true = list(y_true.values())

    print(file_name +
          "\tn=" + str(n) +
          "\tm=" + str(m) +
          "\tnum_clusters=" + str(num_clusters) +
          "\toriginal_modularity=" + str(original_modularity))


    number_of_random_tests = 10
    initial_random_seed = 18081978

    if do_KM1:
        algorithm = "KM1"
        print("Experimenting with", algorithm)
        random_seed = initial_random_seed
        for i in range(number_of_random_tests):
            print("\tRun ", i)
            print(algorithm + " begins.")
            try:
                start = time.time()
                nodeToCluster, clusters = tc.k_means(G, k=num_clusters, randomSeed=random_seed)
                end = time.time()
                elapsed = end - start
                print(algorithm + " ends.")
                y_pred = list(nodeToCluster.values())
                nmi = mt.normalized_mutual_info_score(y_true, y_pred)
                ami = mt.adjusted_mutual_info_score(y_true, y_pred)
                ari = mt.adjusted_rand_score(y_true, y_pred)
                modularity = community_louvain.modularity(nodeToCluster, G)
                num_clusters_found = num_clusters
                write_a_result(file_object, file_name, random_seed, n, m, num_clusters, original_modularity,
                               algorithm, num_clusters_found, modularity, nmi, ami, ari, elapsed)
            except:
                print("Error. Skipping softly...")
                pass
            random_seed += 1
            file_object.flush()

    if do_BKM1:
        algorithm = "BKM1"
        print("Experimenting with", algorithm)
        random_seed = initial_random_seed
        for i in range(number_of_random_tests):
            print("\tRun ", i)
            print(algorithm + " begins.")
            try:
                start = time.time()
                clusters = tc.bisecting_k_means(G, k=num_clusters, randomSeed=random_seed)
                end = time.time()
                elapsed = end - start
                print(algorithm + " ends.")
                nodeToCluster = partition_to_clustering(G, clusters)
                y_pred = list(nodeToCluster.values())
                nmi = mt.normalized_mutual_info_score(y_true, y_pred)
                ami = mt.adjusted_mutual_info_score(y_true, y_pred)
                ari = mt.adjusted_rand_score(y_true, y_pred)
                modularity = community_louvain.modularity(nodeToCluster, G)
                num_clusters_found = num_clusters
                write_a_result(file_object, file_name, random_seed, n, m, num_clusters, original_modularity,
                               algorithm, num_clusters_found, modularity, nmi, ami, ari, elapsed)
            except:
                print("Error. Skipping softly...")
                pass
            random_seed += 1
            file_object.flush()

    if do_GMO and n < 100000:
        algorithm = "GMO"
        print("Experimenting with", algorithm)
        random_seed = initial_random_seed
        for i in range(number_of_random_tests):
            print("\tRun ", i)
            print(algorithm + " begins.")
            start = time.time()
            clusters = community.greedy_modularity_communities(G)
            end = time.time()
            elapsed = end - start
            print(algorithm + " ends.")
            nodeToCluster = partition_to_clustering(G, clusters)
            y_pred = list(nodeToCluster.values())
            nmi = mt.normalized_mutual_info_score(y_true, y_pred)
            ami = mt.adjusted_mutual_info_score(y_true, y_pred)
            ari = mt.adjusted_rand_score(y_true, y_pred)
            modularity = community_louvain.modularity(nodeToCluster, G)
            num_clusters_found = len(set(nodeToCluster.values()))
            write_a_result(file_object, file_name, random_seed, n, m, num_clusters, original_modularity,
                           algorithm, num_clusters_found, modularity, nmi, ami, ari, elapsed)
            random_seed += 1
            file_object.flush()

    if do_LPA:
        algorithm = "LPA"
        print("Experimenting with", algorithm)
        random_seed = initial_random_seed
        for i in range(number_of_random_tests):
            print("\tRun ", i)
            print(algorithm + " begins.")
            CTRL_C_oldu = False
            start = time.time()

            communities_generator = community.label_propagation_communities(G)
            
            clusters = []
            for comm in communities_generator:
                clusters.append(set(comm))
            
            end = time.time()
            elapsed = end - start
            print(algorithm + " ends.")
            if len(clusters) <= 1:
                nmi = 0
                ami = 0
                ari = 0
                modularity = 0
                num_clusters_found = 1
            else:
                nodeToCluster = partition_to_clustering(G, clusters)
                y_pred = list(nodeToCluster.values())
                nmi = mt.normalized_mutual_info_score(y_true, y_pred)
                ami = mt.adjusted_mutual_info_score(y_true, y_pred)
                ari = mt.adjusted_rand_score(y_true, y_pred)
                modularity = community_louvain.modularity(nodeToCluster, G)
                num_clusters_found = len(set(nodeToCluster.values()))

            write_a_result(file_object, file_name, random_seed, n, m, num_clusters, original_modularity,
                           algorithm, num_clusters_found, modularity, nmi, ami, ari, elapsed)
            random_seed += 1
            file_object.flush()


    if do_LVN:
        algorithm = "LVN"
        print("Experimenting with", algorithm)
        random_seed = initial_random_seed
        for i in range(number_of_random_tests):
            print("\tRun ", i)
            print(algorithm + " begins.")
            start = time.time()
            nodeToCluster = community_louvain.best_partition(G, random_state=random_seed)
            end = time.time()
            elapsed = end - start
            print(algorithm + " ends.")
            y_pred = list(nodeToCluster.values())
            nmi = mt.normalized_mutual_info_score(y_true, y_pred)
            ami = mt.adjusted_mutual_info_score(y_true, y_pred)
            ari = mt.adjusted_rand_score(y_true, y_pred)
            modularity = community_louvain.modularity(nodeToCluster, G)
            num_clusters_found = len(set(nodeToCluster.values()))
            write_a_result(file_object, file_name, random_seed, n, m, num_clusters, original_modularity,
                           algorithm, num_clusters_found, modularity, nmi, ami, ari, elapsed)
            random_seed += 1
            file_object.flush()


    if do_SIMCMR:
        algorithm = "SIMCMR"
        print("Experimenting with", algorithm)
        random_seed = initial_random_seed
        for i in range(number_of_random_tests):
            print("\tRun ", i)
            print(algorithm + " begins.")
            start = time.time()
            clusters, min_C_size = tc.SimCMR(G, random_seed=random_seed, node_traversal_order=tc.NodeTraversalOrder.OrderByDegreeAsc)
            end = time.time()
            elapsed = end - start
            print(algorithm + " ends.")
            nodeToCluster = partition_to_clustering(G, clusters)
            y_pred = list(nodeToCluster.values())
            nmi = mt.normalized_mutual_info_score(y_true, y_pred)
            ami = mt.adjusted_mutual_info_score(y_true, y_pred)
            ari = mt.adjusted_rand_score(y_true, y_pred)
            modularity = community_louvain.modularity(nodeToCluster, G)
            num_clusters_found = len(clusters)
            write_a_result(file_object, file_name, random_seed, n, m, num_clusters, original_modularity,
                           algorithm, num_clusters_found, modularity, nmi, ami, ari, elapsed)
            random_seed += 1
            file_object.flush()


    if do_INFOMAP:
        algorithm = "INFOMAP"
        print("Experimenting with", algorithm)
        random_seed = initial_random_seed
        for i in range(number_of_random_tests):
            print("\tRun ", i)
            print(algorithm + " begins.")
            start = time.time()
            
            im = infomap.Infomap("--two-level")
            # print("Building Infomap network from a NetworkX graph...")
            for source, target in G.edges:
                im.add_link(source, target)

            #print("Find communities with Infomap...")
            im.run("--seed " + str(random_seed))
            communities = im.get_modules()
            end = time.time()
            elapsed = end - start

            # IM'den dönen cımmunity id'ler 0'dan değil 1'den başlıyor. 1 azaltmamız gerekiyor uyumluluk için..
            for node, comm in communities.items():
                communities[node] -= 1
            nodeToCluster = communities
            
            
            end = time.time()
            elapsed = end - start
            print(algorithm + " ends.")
            num_clusters_found = len(set(nodeToCluster.values()))
            if num_clusters_found <= 1:
                nmi = 0
                ami = 0
                ari = 0
                modularity = 0
            else:
                # nodeToCluster = vtkm.vt_partition_to_clustering(G, clusters)
                y_pred = list(nodeToCluster.values())
                nmi = mt.normalized_mutual_info_score(y_true, y_pred)
                ami = mt.adjusted_mutual_info_score(y_true, y_pred)
                ari = mt.adjusted_rand_score(y_true, y_pred)
                modularity = community_louvain.modularity(nodeToCluster, G)
                

            write_a_result(file_object, file_name, random_seed, n, m, num_clusters, original_modularity,
                           algorithm, num_clusters_found, modularity, nmi, ami, ari, elapsed)
            random_seed += 1
            file_object.flush()


    if do_OSLOM:
        algorithm = "OSLOM"
        print("Experimenting with", algorithm)
        random_seed = initial_random_seed
        for i in range(number_of_random_tests):
            print("\tRun ", i)
            print(algorithm + " begins.")
            
            args = Namespace()
            args.min_cluster_size = 0
            args.oslom_exec = r"d:\\oslom_undir.exe" # oslom.DEF_OSLOM_EXEC
            args.oslom_args = oslom_runner.DEF_OSLOM_ARGS
            args.oslom_args.append("-seed")
            args.oslom_args.append(str(random_seed))
    
            edges = []
            edges = [(u, v, 1) for u,v in G.edges]

            start = time.time()

            clusters = oslom_runner.run_in_memory(args, edges)

            end = time.time()
            elapsed = end - start

            temp_node_to_cluster = {}

            for cluster in clusters[0]['clusters']:
                cluster_id = cluster['id']
                nodes = cluster['nodes'] # list
                for node_dict in nodes:
                    node_id = node_dict['id']
                    temp_node_to_cluster[node_id] = cluster_id

            # node_to_cluster'ı sıralamak lazım
            node_to_cluster = {}
            for node in G:
                node_to_cluster[node] = temp_node_to_cluster[node]

            elapsed = clusters[2]  # işlem süresini buradan almak daha doğru olur...
            nodeToCluster = node_to_cluster
            
            end = time.time()
            elapsed = end - start
            print(algorithm + " ends.")
            num_clusters_found = len(set(nodeToCluster.values()))
            if num_clusters_found <= 1:
                nmi = 0
                ami = 0
                ari = 0
                modularity = 0
            else:
                # nodeToCluster = vtkm.vt_partition_to_clustering(G, clusters)
                y_pred = list(nodeToCluster.values())
                nmi = mt.normalized_mutual_info_score(y_true, y_pred)
                ami = mt.adjusted_mutual_info_score(y_true, y_pred)
                ari = mt.adjusted_rand_score(y_true, y_pred)
                modularity = community_louvain.modularity(nodeToCluster, G)
                

            write_a_result(file_object, file_name, random_seed, n, m, num_clusters, original_modularity,
                           algorithm, num_clusters_found, modularity, nmi, ami, ari, elapsed)
            random_seed += 1
            file_object.flush()


    file_object.close()

