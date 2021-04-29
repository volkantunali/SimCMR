# -*- coding: utf-8 -*-

#import networkx as nx
import time
import numpy as np
import os
import subprocess
import shlex

def create_network_name(n, tau1, tau2, mu, average_degree, max_degree, min_community, max_community, i):
    network_name = "n_" + str(n) + \
                   "_t1_" + str(tau1) + \
                   "_t2_" + str(tau2) + \
                   "_mu_" + str(mu) + \
                   "_k_" + str(average_degree) + \
                   "_maxk_" + str(max_degree) + \
                   "_minc_" + str(min_community) + \
                   "_maxc_" + str(max_community) + \
                   "__" + str(i)
    return network_name

def create_networks(n, tau1, tau2, mu, average_degree, max_degree, min_community, max_community, folder="", num_graphs=10):
#    random_seeds = set([])
    for i in range(num_graphs):
        start = time.time()
#        while True:
#            random_seed=np.random.randint(1, 1000000)
#            if random_seed not in random_seeds:
#                random_seeds.add(random_seed)
#                break

#        if os.path.exists("time_seed.dat"):
#            os.remove("time_seed.dat")

        network_name = create_network_name(n, tau1, tau2, mu, average_degree, max_degree, min_community, max_community, i)

        print("")
        print("trying to generate " + network_name + "...")

#                G = nx.algorithms.community.LFR_benchmark_graph(n, tau1, tau2, mu,
#                                                                average_degree=average_degree,
#                                                                min_degree=None,
#                                                                max_degree=max_degree,
#                                                                min_community=min_community,
#                                                                max_community=max_community,
#                                                                max_iters=500, seed=random_seed)


        lfr_params = "D:\\lfr_benchmark.exe -N " + str(n) + " -k " + str(average_degree) + " -maxk " + str(max_degree) + \
                     " -mu " + str(mu) + " -minc " + str(min_community) + " -maxc " + str(max_community) + \
                     " -t1 " + str(tau1) + " -t2 " + str(tau2)

        print("lfr params: ", lfr_params)
        subprocess.run(shlex.split(lfr_params))

        end = time.time()
        elapsed = end - start
        print(network_name + " created in " + str(elapsed) + " seconds.")


        with open("time_seed.dat", "r") as f:
            random_seed = f.read()
            random_seed = random_seed.strip()
            f.close()

        network_name += "__" + random_seed + "__" + str(elapsed)

        file_name = network_name + ".network.txt"
        os.rename("network.dat", file_name)

        file_name = network_name + ".community.txt"
        os.rename("community.dat", file_name)

        file_name = network_name + ".statistics.txt"
        os.rename("statistics.dat", file_name)

        file_name = network_name + ".time_seed.txt"
        with open(file_name, "w") as f:
            f.write(random_seed)
            f.close()

    return


mu_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

n = 1000
tau1 = 2  # GAMMA [minus exponent for the degree sequence]
tau2 = 1  # BETA [minus exponent for the community size distribution]
mu = 0.1 # MU
average_degree=25
min_degree=None
max_degree=int(n/10)
min_community = 50
max_community = int(n/10)

for mu in mu_list:
    create_networks(n, tau1, tau2, mu, average_degree, max_degree, min_community, max_community, "Graphs/1000/")

"""
n = 10000
tau1 = 2  # GAMMA [minus exponent for the degree sequence]
tau2 = 1  # BETA [minus exponent for the community size distribution]
mu = 0.1 # MU
average_degree=25
min_degree=None
max_degree=int(n/10)
min_community = 50
max_community = int(n/10)

for mu in mu_list:
    create_networks(n, tau1, tau2, mu, average_degree, max_degree, min_community, max_community, "Graphs/10000/")


n = 100000
tau1 = 2  # GAMMA [minus exponent for the degree sequence]
tau2 = 1  # BETA [minus exponent for the community size distribution]
mu = 0.1 # MU
average_degree=25
min_degree=None
max_degree=int(n/10)
min_community = 50
max_community = int(n/10)

for mu in mu_list:
    create_networks(n, tau1, tau2, mu, average_degree, max_degree, min_community, max_community, "Graphs/100000/")


mu_list = [0.1]

# Bu boyutta her mu'den 1 adet üretelim
n = 1000000
tau1 = 2  # GAMMA [minus exponent for the degree sequence]
tau2 = 1  # BETA [minus exponent for the community size distribution]
mu = 0.1 # MU
average_degree=25
min_degree=None
max_degree=int(n/10)
min_community = int(n/100)
max_community = int(n/10)

for mu in mu_list:
    create_networks(n, tau1, tau2, mu, average_degree, max_degree, min_community, max_community, "Graphs/1000000/")

"""
