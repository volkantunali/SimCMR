# -*- coding: utf-8 -*-

"""
This module implements SimCMR network community detection
algorithm developed by Volkan TUNALI (2020). SimCMR currently
works on undirected and unweighted networks.

Although, SimCMR is initially implemented on NetworkX framework,
it can easily be modified to work with any graph framework as
SimCMR requires only neighborhood information of nodes, and
node degrees. There is no dependency to specific functions of
Networkx package.

In addition, K-Means and Bisecting K-Means algorithms that
work on graphs to detect communities are also implemented here.
Similar to SimCMR, K-Means can be modified to work with other
graph frameworks with great ease. Unlike SimCMR, however,
Bisecting K-Means is dependent on some NetworkX functions like
subgraph. It may require some work to modify it to work with
other graph frameworks.

"""


import networkx as nx
import numpy as np
import enum

__author__ = """Volkan TUNALI (volkan.tunali@gmail.com)"""


class NodeTraversalOrder(enum.Enum):
   NaturalNodeOrder= 1
   OrderByDegreeAsc = 2
   OrderByDegreeDesc = 3
   RandomOrder = 4


def SimCMR(G,
           min_cluster_size=None,
           min_cluster_size_factor=1.,
           node_traversal_order=NodeTraversalOrder.OrderByDegreeAsc,
           phase_no_to_return=0,
           random_seed=None):

    if min_cluster_size_factor <= 0:
        raise ValueError("min_cluster_size_factor muster be grater than zero.")

    # We store neighbor data for really faster access than NetworkX's own functions.
    G_neighbors = {}
    for node in G:
        G_neighbors[node] = set(G._adj[node])

    # "clusters" has to be a List, not a set because we traverse the clusters in some order later.
    # It will be a list of clusters, and each cluster will be a crisp set of nodes.
    clusters = []


    # PHASE #1 ----------------------------------------------------------------------------------------------
    # We produce initial candidate clusters by a kind of label propagation
    # beginning from the node with the highest degree. This is the fastest
    # part of SimCMR; even millions of nodes are traversed in no time.
    visited_nodes = set()
    nodes_sorted_by_degree_desc = [node for (node, degree) in sorted(G.degree, key=lambda x: x[1], reverse=True)]
    for node in nodes_sorted_by_degree_desc:
        if node not in visited_nodes:
            # If this node is never traversed yet, create a new cluster with it.
            new_cluster = {node}
            clusters.append(new_cluster)
            # Add its all neighbors to the same cluster if they haven't been added to any cluster yet.
            for neighber_of_node in G_neighbors[node]:
                if neighber_of_node not in visited_nodes:
                    new_cluster.add(neighber_of_node)
                    visited_nodes.add(neighber_of_node)
            visited_nodes.add(node)

    del visited_nodes # We no longer need this set, so we simply delete it here.

    if phase_no_to_return == 1:
        return clusters, min_cluster_size


    # PHASE #2 ----------------------------------------------------------------------------------------------
    # We merge small clusters to larger ones based on similarity between nodes and clusters
    # (n-to-c-sim) if they are smaller than our cluster size threshold, which is
    # calculated here based on some heuristics experimentally decided: min_cluster_size
    # This means that, after this phase, there will be no cluster smaller than min_cluster_size.
    if min_cluster_size == None:
        # This is our experimentally found heuristic formula.
        min_cluster_size_heuristics_1 = round((float(G.number_of_nodes()) ** (1./2.75)) - 4, 0)
        cluster_sizes = [len(c) for c in clusters]
        # In some cases, depending on the size distribution of initial candidate clusters,
        # min_cluster_size can be larger than an appropriate value. So, we apply
        # another heuristic based on the size distribution of candidate clusters.
        min_cluster_size_heuristics_2 = float(np.sqrt(np.average(np.power(np.array(cluster_sizes, dtype=np.int64), 2))))
        # Better result is achieved when we use the smallest of both.
        min_cluster_size = min(min_cluster_size_heuristics_1, min_cluster_size_heuristics_2)

    # The client of this method can fine-tune the min_cluster_size by
    # supplying min_cluster_size_factor other than 1.0
    min_cluster_size *= min_cluster_size_factor

    # If any of the heuristics above fails to calculate a min_cluster_size less than 2, then
    # we set it 2 because, a cluster must contain at least 2 nodes for SimCMR.
    if min_cluster_size < 2:
        min_cluster_size = 2

    clusters = sorted(clusters, key=len, reverse=False)
    num_clusters = len(clusters)
    removed_clusters = []
    node_to_cluster = get_node_to_cluster_dict_from_clusters(G, clusters)
    for i in range(num_clusters - 1):
        cluster = clusters[i]
        cluster_size = len(cluster)
        # If this cluster is larger than or equal to min_cluster_size, then
        # we do not try to merge it other larger clusters.
        if cluster_size >= min_cluster_size:
            # Due to node transfers among clusters, some later clusters may have
            # become smaller, so, we do not just break here. Instead, we check all the remaining.
            continue

        # If this is a singleton and it has only one neighbor, then we simply
        # put that single node into its only neighbor's cluster to make things faster.
        if cluster_size == 1:
            # This is the fastest way to get the only element from a set in Python.
            for node in cluster:
                break

            if G.degree[node] == 1:
                # Again, this is the fastest way to extract the only neighbor.
                for neighbor in G_neighbors[node]:
                    break

                cluster_index_of_neighbor = node_to_cluster[neighbor]
                cluster_of_neighbor = clusters[cluster_index_of_neighbor]
                cluster_of_neighbor.add(node)
                removed_clusters.append(cluster)
                node_to_cluster[node] = cluster_index_of_neighbor
                cluster.remove(node)
                continue # Now that we have processed this singleton here totally, we skip to next one.

        # For each node in this cluster, we calculate similarity between that node and
        # other clusters which are further in our clusters list (we do not look back here).
        removed_nodes = []
        for node in cluster:
            max_sim = -999999
            max_sim_cluster = None
            max_sim_cluster_index = -1

            # Instead of traversing all other clusters one by one, we do a simple trick here
            # that we traverse only the clusters where the neighbors of this node are in.
            # Of course, we do not consider the own cluster of the node because here we are
            # trying to get rid of the current cluster by distributing its all nodes to
            # other larger clusters.
            cluster_indexes_of_neighbors = {node_to_cluster[x] for x in G_neighbors[node]}
            for j in cluster_indexes_of_neighbors:
                # We do not consider the own cluster of the node, and also all previous clusters.
                if j <= i:
                    continue

                cluster2 = clusters[j]
                # If this cluster is also a small cluster, we do not consider it for merging.
                if len(cluster2) < min_cluster_size:
                    continue

                # Actual similarity calculation between the node and the next cluster
                sim = calc_node_to_cluster_similarity(G_neighbors, node, cluster2)
                # If it is the largest similarity so far, we note it.
                if sim > max_sim:
                    max_sim = sim
                    max_sim_cluster = cluster2
                    max_sim_cluster_index = j

            # If we could find a more similar cluster for the node, we put it
            # into that cluster. Otherwise, we simply do no change its cluster.
            if max_sim > 0 and max_sim_cluster != None:
                max_sim_cluster.add(node)
                node_to_cluster[node] = max_sim_cluster_index
                removed_nodes.append(node)

        # If all the nodes in current cluster have been transferred into some
        # other clusters, than we mark this cluster to be removed from the global clusters list.
        for node in removed_nodes:
            cluster.remove(node)
        if len(cluster) <= 0:
            removed_clusters.append(cluster)

    # At the end of Phase #2, we get rid of empty clusters.
    for cluster in removed_clusters:
        clusters.remove(cluster)


    if phase_no_to_return == 2:
        return clusters, min_cluster_size


    # PHASE #3 ----------------------------------------------------------------------------------------------
    # We traverse the nodes in the order determined by the node_traversal_method
    # parameter, and put each node into its most similar cluster based on
    # similarity between node and cluster (n-to-c-sim). We do this in 5
    # identical iterations. 5 is an optimal value decided experimentally.
    # More iterations could produce better results but slows down the whole process.
    node_to_cluster = get_node_to_cluster_dict_from_clusters(G, clusters)

    if node_traversal_order == NodeTraversalOrder.NaturalNodeOrder:
        del nodes_sorted_by_degree_desc
        node_list_for_traversal = list(G)
    elif node_traversal_order == NodeTraversalOrder.OrderByDegreeAsc: # This is the default of SimCMR
        del nodes_sorted_by_degree_desc
        node_list_for_traversal = [node for (node, degree) in sorted(G.degree, key=lambda x: x[1], reverse=False)]
    elif node_traversal_order == NodeTraversalOrder.OrderByDegreeDesc:
        node_list_for_traversal = nodes_sorted_by_degree_desc # We already have this list from Phase #1.
    else:
        del nodes_sorted_by_degree_desc
        random_state = np.random.RandomState(random_seed)
        node_list_for_traversal = random_state.permutation(list(G))

    # Here we employ a heuristic that if a node has so many neighbors within the
    # same cluster with itself, then, we do not look for better clusters to move in.
    # Therefore, we can speedup this phase dramatically. The most intuitive threshold
    # for this "so many neighbors" is 50% in general. However, in some difficult
    # cases (like LFR benchmarks with large mu, for example), there is a huge
    # candidate cluster where almost all nodes take place. In such cases, 50%
    # threshold needs to be increased. We calculate this threshold by dividing
    # the size of the largest cluster by N and multiply by it 1.1 to have some
    # tolerance. However, if this threshold is greater than 66%, we limit it to 66%.
    largest_cluster_node_count = 0
    for cluster in clusters:
        l = len(cluster)
        if l > largest_cluster_node_count:
            largest_cluster_node_count = l

    neighbor_ratio_threshold = float(largest_cluster_node_count) / float(G.number_of_nodes())
    neighbor_ratio_threshold *= 1.1;
    if neighbor_ratio_threshold > 0.66:
        neighbor_ratio_threshold = 0.66
    if neighbor_ratio_threshold < 0.5:
        neighbor_ratio_threshold = 0.5

    # 5 iterations is always good and fast.
    max_iter = 5
    removed_clusters = []
    for i in range(max_iter):
        number_of_cluster_changes = 0
        for node in node_list_for_traversal:
            own_cluster_of_node = clusters[node_to_cluster[node]]
            neighbor_ratio = calc_ratio_of_neighbors_of_node_in_cluster(G_neighbors, node, own_cluster_of_node)
            if neighbor_ratio < neighbor_ratio_threshold:
                max_sim = -999999
                max_sim_cluster_index = None

                cluster_indexes_of_neighbors = {node_to_cluster[x] for x in G_neighbors[node]}
                # We calculate the similarity to the own cluster of the node here, so we add it to the list.
                cluster_indexes_of_neighbors.add(node_to_cluster[node])

                # We calculate similarity between the node and the clusters, including
                # the node's own cluster.
                for i in cluster_indexes_of_neighbors:
                    cluster = clusters[i]
                    # If the cluster is small and it is other than the node's own cluster,
                    # we do not consider it for similarity calculation.
                    if len(cluster) < min_cluster_size and node_to_cluster[node] != i:
                        continue

                    sim = calc_node_to_cluster_similarity(G_neighbors, node, cluster)
                    if sim > max_sim:
                        max_sim = sim
                        max_sim_cluster = cluster
                        max_sim_cluster_index = i

                old_cluster_index_of_node = node_to_cluster[node]
                if max_sim > 0 and max_sim_cluster_index != old_cluster_index_of_node:
                    old_cluster = clusters[old_cluster_index_of_node]
                    max_sim_cluster.add(node)
                    old_cluster.remove(node)
                    node_to_cluster[node] = max_sim_cluster_index
                    if len(old_cluster) <= 0:
                        removed_clusters.append(old_cluster)
                    number_of_cluster_changes += 1

        # If no more cluster change, no need to iterate until 5 iterations.
        if number_of_cluster_changes <= 1:
            break;

    for cluster in removed_clusters:
        clusters.remove(cluster)


    if phase_no_to_return == 3:
        return clusters, min_cluster_size


    # PHASE #4 ----------------------------------------------------------------------------------------------
    # We merge small clusters to larger ones based on similarity between clusters
    # (c-to-c-sim). When we merge two clusters based on their similarity, we
    # check if this merge has improved the modularity or not. If it has, we
    # make the merge persistent. If not, we do not merge them, and move on to
    # the next small cluster.
    clusters = sorted(clusters, key=len, reverse=False)
    # These are used to make modularity computation very fast.
    cluster_degrees = dict() # by cluster index
    cluster_in_degrees = dict() # by cluster index
    node_to_cluster = get_node_to_cluster_dict_from_clusters(G, clusters) # by node id
    num_clusters = len(clusters)
    total_weight = G.size()
    cluster_to_cluster_link_count = np.zeros((num_clusters, num_clusters), dtype=np.int32)
    modularity = calc_initial_modularity_status(G, # in
                                                G_neighbors, # in
                                                clusters, # in
                                                node_to_cluster, # in
                                                cluster_degrees, # out
                                                cluster_in_degrees, # out
                                                cluster_to_cluster_link_count, # out
                                                total_weight) #in

    # Current modularity is the best modularity we have so far.
    best_modularity = modularity

    for i in range(num_clusters-1):
        cluster1 = clusters[i]
        if cluster1 == None:
            continue
        max_sim_cluster = None
        max_sim = -99999
        max_sim_cluster_index = -1

        # We find the most similar cluster to cluster1 and
        # try to merge cluster1 to it if it improves modularity.
        for j in range(i+1, num_clusters):
            cluster2 = clusters[j]
            if cluster2 == None or len(cluster2) <= 0:
                continue

            sim = calc_cluster_to_cluster_similarity(G, cluster1, cluster2,
                                                     i, j, cluster_to_cluster_link_count)
            if sim > max_sim:
                max_sim_cluster = cluster2
                max_sim = sim
                max_sim_cluster_index = j

        if max_sim > 0 and max_sim_cluster != None:
            j = max_sim_cluster_index
            cluster2 = max_sim_cluster

            # We backup some values in case we do not decide to really
            # merge the clusters.
            backup_modularity = modularity
            backup_cluster_degrees = cluster_degrees.get(j, 0.)
            backup_cluster_in_degrees = cluster_in_degrees.get(j, 0.)

            # We update the values of the second cluster
            # because we add cluster1 to it to make it larger.
            cluster_degrees[j] = cluster_degrees.get(j, 0.) + \
                                 cluster_degrees.get(i, 0.)

            cluster_in_degrees[j] = cluster_in_degrees.get(j, 0.) + \
                                    cluster_in_degrees.get(i, 0.) + \
                                    cluster_to_cluster_link_count[i, j]

            backup_node_to_cluster = dict()
            for node in cluster1:
                backup_node_to_cluster[node] = i # Save the old cluster index
                node_to_cluster[node] = j # Update them with the new cluster index

            # Using updated values, we calculate new modularity with the fastest way possible.
            modularity = calc_modularity(node_to_cluster, cluster_degrees, cluster_in_degrees, total_weight)

            # If this merge has not increased modularity, we restore everything back
            # to the previous state using the backup variables easily and fast.
            if modularity <= best_modularity:
                modularity = backup_modularity
                cluster_degrees[j] = backup_cluster_degrees
                cluster_in_degrees[j] = backup_cluster_in_degrees
                for node in backup_node_to_cluster:
                     node_to_cluster[node] = backup_node_to_cluster[node]
            else:
                # If modularity has improved, we do the real updates.
                # Move all nodes in cluster1 into cluster2.
                cluster2.update(cluster1)
                # Clear cluster1.
                cluster1.clear() #küme1'de eleman kalmasın

                # Update cluster_to_cluster_link_count using the values regarding cluster1.
                # This is the effect of the merging.
                for k in range(i+1, j):
                    cluster_to_cluster_link_count[k, j] += cluster_to_cluster_link_count[k, i]

                # Save this new modularity as the best one.
                best_modularity = modularity
                # There is no longer a cluster at index i.
                clusters[i] = None


    # We update our clusters list with the clusters other than None from the previous merges.
    clusters = [c for c in clusters if c != None]


    if phase_no_to_return == 4:
        return clusters, min_cluster_size


    # PHASE #5 ----------------------------------------------------------------------------------------------
    # This is almost the same process as Phase #3 with minor differences.
    # First, we use 50% constant value for neighbor_ratio_threshold.
    # Second, we do not check the size of the clusters because at this
    # phase, all clusters have been detected, and their sizes should be
    # at least min_cluster_size.
    node_to_cluster = get_node_to_cluster_dict_from_clusters(G, clusters)
    # 5 iterations is always good and fast.
    max_iter = 5
    neighbor_ratio_threshold = 0.5
    removed_clusters = []
    for i in range(max_iter):
        number_of_cluster_changes = 0
        for node in node_list_for_traversal:
            own_cluster_of_node = clusters[node_to_cluster[node]]
            neighbor_ratio  = calc_ratio_of_neighbors_of_node_in_cluster(G_neighbors, node, own_cluster_of_node)
            if neighbor_ratio < neighbor_ratio_threshold:
                max_sim = -999999
                max_sim_cluster_index = None

                cluster_indexes_of_neighbors = {node_to_cluster[x] for x in G_neighbors[node]}
                # We calculate the similarity to the own cluster of the node here, so we add it to the list.
                cluster_indexes_of_neighbors.add(node_to_cluster[node])

                # We calculate similarity between the node and the clusters, including
                # the node's own cluster.
                for i in cluster_indexes_of_neighbors:
                    cluster = clusters[i]
                    sim = calc_node_to_cluster_similarity(G_neighbors, node, cluster)
                    if sim > max_sim:
                        max_sim = sim
                        max_sim_cluster = cluster
                        max_sim_cluster_index = i

                old_cluster_index_of_node = node_to_cluster[node]
                if max_sim > 0 and max_sim_cluster_index != old_cluster_index_of_node:
                    old_cluster = clusters[old_cluster_index_of_node]
                    max_sim_cluster.add(node)
                    old_cluster.remove(node)
                    node_to_cluster[node] = max_sim_cluster_index
                    if len(old_cluster) <= 0:
                        removed_clusters.append(old_cluster)
                    number_of_cluster_changes += 1

        # If no more cluster change, no need to iterate until 5 iterations.
        if number_of_cluster_changes <= 1:
            break;

    for cluster in removed_clusters:
        clusters.remove(cluster)


    return clusters, min_cluster_size
# End of SimCMR()


def calc_initial_modularity_status(G, # in
                                   G_neighbors, # in
                                   clusters, # in
                                   node_to_cluster, # in
                                   cluster_degrees, # out
                                   cluster_in_degrees, # out
                                   cluster_to_cluster_link_count, # out
                                   total_weight):
    for node in G:
        com = node_to_cluster[node]
        cluster_degrees[com] = cluster_degrees.get(com, 0.) + G.degree(node)
        for neighbor in G_neighbors[node]:
            neighbor_com = node_to_cluster[neighbor]
            if node != neighbor:
                cluster_to_cluster_link_count[com, neighbor_com] += 1

            if neighbor_com == com:
                if neighbor == node:
                    cluster_in_degrees[com] = cluster_in_degrees.get(com, 0.) + 1
                else:
                    cluster_in_degrees[com] = cluster_in_degrees.get(com, 0.) + 0.5

    mod = 0. # modularity
    for com in set(node_to_cluster.values()):
        mod += (cluster_in_degrees.get(com, 0.) / total_weight) - \
               (cluster_degrees.get(com, 0.) / (2. * total_weight)) ** 2
    return mod



def calc_modularity(node_to_cluster, cluster_degrees, cluster_in_degrees, total_weight):
    """
    Fast compute the modularity of the partition of the graph using
    status precomputed
    """
    links = float(total_weight)
    result = 0.
    for community in set(node_to_cluster.values()):
        in_degree = cluster_in_degrees.get(community, 0.)
        degree = cluster_degrees.get(community, 0.)
        result += in_degree / links - ((degree / (2. * links)) ** 2)
    return result



def get_node_to_cluster_dict_from_clusters(G, clusters):
    # We want this dict to have items in exactly the same order
    # of the nodes in G because we also use this to calculate
    # clustering validity indices like NMI. So, order of
    # the items are important. That's why we don't simply
    # traverse the clusters in one pass.
    node_to_cluster = {}

    temp = {}
    # Cluster indices must start from zero.
    for (i, cluster) in enumerate(clusters, 0):
        for node in cluster:
            temp[node] = i

    for node in G:
        node_to_cluster[node] = temp[node]

    return node_to_cluster



def calc_node_to_cluster_similarity(G_neighbors, node, cluster):
    c = len(cluster)

    # If the node is also in the same cluster, we discount it by 1.
    if node in cluster:
        c = c - 1

    if c == 0:
        return 0

    # To make the intersection faster in Python, we do a simple trick here.
    neighbors = G_neighbors[node]
    if len(neighbors) < c:
        return len(neighbors.intersection(cluster)) / c

    return len(cluster.intersection(neighbors)) / c



def calc_cluster_to_cluster_similarity(G,
                                       cluster1, cluster2,
                                       cluster1_index, cluster2_index,
                                       cluster_to_cluster_link_count):
    # This is a very fast way of computing similarity between clusters
    # using the precomputed and stored values in cluster_to_cluster_link_count.
    return cluster_to_cluster_link_count[cluster1_index, cluster2_index]  / (len(cluster1) + len(cluster2))



def calc_ratio_of_neighbors_of_node_in_cluster(G_neighbors, node, cluster):
    set_of_neighbors = G_neighbors[node]
    l = len(set_of_neighbors)
    if l == 0:
        return 0

    # To make the intersection faster in Python, we do a simple trick here.
    if l < len(cluster):
        return len(set_of_neighbors.intersection(cluster)) / l

    return len(cluster.intersection(set_of_neighbors)) / l


# ------------------------------------------------------------------------------------------------------------
# ----- K-Means and Bisecting K-Means implementations --------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def k_means(G, k, randomSeed=None, online_update=True):
    # max number of k-means loop iterations if no convergence is achieved before.
    maxIter = 25

    # each cluster is a set of nodes
    clusters = [set([]) for x in range(k)]
    nodeToCluster = {}

    n = G.number_of_nodes()

    selectedNodes = list(G.nodes)
    if randomSeed != None:
        np.random.seed(randomSeed)
    randomClusters = np.random.randint(0, k, n)

    # Assign each node into randomly selected cluster.
    for i in range(n):
        clusterIndex = randomClusters[i]
        currentNode = selectedNodes[i]
        clusters[clusterIndex].add(currentNode)
        nodeToCluster[currentNode] = clusterIndex


    objFunc = 0.
    prevObjFunc = 0.
    bestNodeToCluster = {}
    bestClusters = []
    for iter in range(1, maxIter):
        objFunc = 0.

        if online_update:
            # For each node find the closest cluster...
            for node in G:
                clusterIndexOfThisNode = nodeToCluster[node]
                clusterOfThisNode = clusters[clusterIndexOfThisNode]
                clusterOfThisNode.remove(node)

                # Find the most similar cluster to the node.
                maxSimilarity = -9999999
                maxSimilarityClusterNo = -1
                for i in range(k):
                    cluster = clusters[i]
                    similarity = calc_node_to_cluster_similarity_for_k_means(G, node, cluster)
                    if similarity > maxSimilarity:
                        maxSimilarity = similarity
                        maxSimilarityClusterNo = i

                nodeToCluster[node] = maxSimilarityClusterNo
                cluster = clusters[maxSimilarityClusterNo]
                cluster.add(node)
                objFunc += maxSimilarity
        else:
            new_nodeToCluster = {}
            for node in G:
                # Find the most similar cluster to the node.
                maxSimilarity = -9999999
                maxSimilarityClusterNo = -1
                for i in range(k):
                    cluster = clusters[i]
                    similarity = calc_node_to_cluster_similarity_for_k_means(G, node, cluster)
                    if similarity > maxSimilarity:
                        maxSimilarity = similarity
                        maxSimilarityClusterNo = i

                new_nodeToCluster[node] = maxSimilarityClusterNo
                objFunc += maxSimilarity


            for (n,c) in new_nodeToCluster.items():
                old_c = nodeToCluster[n]
                if c != old_c:
                    old_cluster = clusters[old_c]
                    old_cluster.remove(n)

                    new_cluster = clusters[c]
                    new_cluster.add(n)

                    nodeToCluster[n] = c


        # if no further improvement, then accept the local optimum and break the main loop
        if abs(objFunc - prevObjFunc) < 0.00001:
            break

        if objFunc < prevObjFunc:
            break


        prevObjFunc = objFunc
        bestNodeToCluster = nodeToCluster.copy()
        bestClusters = [cluster.copy() for cluster in clusters]

    return bestNodeToCluster, bestClusters



def calc_node_to_cluster_similarity_for_k_means(G, node, cluster):
    c = len(cluster)

    if node in cluster:
        c = c - 1

    if c == 0:
        return 0

    # To make the intersection faster in Python, we do a simple trick here.
    neighbors = set(G.neighbors(node))
    if len(neighbors) < c:
        return len(neighbors.intersection(cluster)) / c

    return len(cluster.intersection(neighbors)) / c


def bisecting_k_means(G, k, randomSeed=None):
    # Initially, we have a single cluster that includes the whole G.
    graphs = set([G])

    _k = 1
    while _k < k:
        _k += 1

        # We select a graph to bisect depending on the node count.
        max_nodes_graph = None
        max_nodes_count = 0
        for g in graphs:
            if len(g.nodes) > max_nodes_count:
                max_nodes_count = len(g.nodes)
                max_nodes_graph = g

        selected_G = max_nodes_graph

        # Remove the selected graph from our current graphs set.
        graphs.remove(selected_G);

        # Find 2 clusters inside the selected graph. That is, k is always 2.
        nodeToCluster, clusters = k_means(selected_G, 2, randomSeed=randomSeed)

        cluster_1 = clusters[0]
        cluster_2 = clusters[1]
        G1 = nx.subgraph(G, cluster_1).copy()
        G2 = nx.subgraph(G, cluster_2).copy()

        graphs.add(G1)
        graphs.add(G2)

    # The main iteration is over. Now, we have k graphs.
    # We return clusters that include their nodes.
    clusters = [set([]) for x in range(k)]
    i = 0
    for g in graphs:
        for n in g:
            clusters[i].add(n)
        i += 1


    return clusters

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

