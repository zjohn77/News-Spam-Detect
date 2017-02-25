import numpy as np
import pandas as pd

class Silhouette(object):
    '''Define class for creating and processing a list of silhouettes values from each input.'''
    def __init__(self, doc_term_matrix_, clustered_sentences_):
        self.__doc_term_matrix_ = doc_term_matrix_
        self.__clustered_sentences_ = clustered_sentences_
        self.__sample_silhouettes = silhouette_samples(X = doc_term_matrix_, labels = clustered_sentences_)
        # List of the silhouette of each input data point.

    def avg_list(self, sample_):
        '''calc avg score of each cluster'''
        return sum(sample_)/len(sample_)
    # This results in division by 0 sometimes!!!
        
    def select_cluster(self, clusID_):
        '''pick each cluster in the silhouettes vector'''
        return self.__sample_silhouettes[self.__clustered_sentences_ == clusID_]        
            
    def averaged_sils_by_cluster(self, num_clusters_):
        '''Aggregate the silhouette scores for samples belonging to cluster i'''    
        avg_clus_silhouette = pd.Series(index = range(num_clusters_)) # initialize Series to hold avg Sils          

        for clusID in xrange(num_clusters_):
            ith_cluster = self.select_cluster(clusID)
            avg_clus_silhouette[clusID] = self.avg_list(ith_cluster)
        return avg_clus_silhouette.sort_values(ascending = False)    