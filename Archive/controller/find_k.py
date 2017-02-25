import numpy as np
import pandas as pd

def find_best_k(X, K_CLUSTERS = 10, dist_measure = cosine_distance, repeats = 10):  
    '''Determine optimal k parameter by iterating over k from (2, K_CLUSTERS], and comparing avg 
    Silhouette Score of each cluster configuration.'''    
    
    average_silhouette_list = [None] * (K_CLUSTERS - 1) # Store silhouette of each K tried in a list.        
    
    for k in xrange(2, K_CLUSTERS + 1):  # Loop from 2 to max K to try.       
        KSelector = KMeansClusterer(num_means = k
                                    , distance = dist_measure
                                    , repeats = repeats
                                    , avoid_empty_clusters = True) # Instantiate a KMeans object.
        
        cluster_labels = np.array(KSelector.cluster(X, assign_clusters=True)) # Run KMeans algo on matrix X.
        
        # Calc and store to list the average silhouette score for KMeans on this number of clusters.
        average_silhouette_list[k-2] = silhouette_score(X, labels = cluster_labels)                            
    
    return 2 + average_silhouette_list.index(max(average_silhouette_list)) # return the best K found


best_k = find_best_k(doc_term_matrix.toarray()
                     , K_CLUSTERS = 20
                     , dist_measure = cosine_distance
                     , repeats = 5)