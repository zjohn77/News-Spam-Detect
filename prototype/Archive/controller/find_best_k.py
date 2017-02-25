import numpy as np
import pandas as pd
from itertools import product


################################################################
# 2. First, Parameter Grid Search: to discover best K (and corresponding avg silhouette) 
# for each parameter configuration on a devised Parameter Grid. 
# Second, store entire silhouette array for each parameter config into a dict.
################################################################
# Create an iterable of grid points for parameter combinations I want to try.
distance = [cosine_distance]
repeats = [5, 10]
parameter_grid = product(distance, repeats)



# Compute the best K for each element of parameter grid, and store results in a dict keyed by parameters from grid.
K = 20
grid_k_dict = {key: find_best_k(doc_term_matrix.toarray()
                                 , K_CLUSTERS = K
                                 , dist_measure = key[0]
                                 , repeats = key[1]) for key in parameter_grid}

cluster_sils_by_param = dict.fromkeys(grid_k_dict)

for config in grid_k_dict:
    # Cluster with best K of each grid point; then store its list of silhouettes into the list 'cluster_average_sils'. 
    Clusterer = KMeansClusterer(num_means = grid_k_dict[config]
                                , distance = config[0]
                                , repeats = config[1]
                                , avoid_empty_clusters = True) # Instantiate a KMeans object.  
    
    sil = Silhouette(doc_term_matrix
                     , np.array(Clusterer.cluster(doc_term_matrix.toarray(), assign_clusters = True)))

    # Put list of avg cluster sils into a dict keyed by param grid. 
    cluster_sils_by_param[config] = sil.averaged_sils_by_cluster(Clusterer.num_clusters())     