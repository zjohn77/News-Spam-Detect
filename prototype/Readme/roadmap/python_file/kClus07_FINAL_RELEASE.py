
# coding: utf-8

# In[1]:

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.cluster import KMeansClusterer, cosine_distance, euclidean_distance
from nltk.metrics.distance import jaccard_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np

import collections
from pprint import pprint
import string
import re
from itertools import product


# In[3]:

################################################################
# 0. Define Functions and Classes
################################################################
class LemmaTokenizer(object):
    '''Define class for processing raw string into 1-gram lemmas exclusive of stop-words.'''
    def __init__(self):
        self.wnl = WordNetLemmatizer() # instantiate a Lemmatizer object
    def __call__(self, doc):
        lemmas = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
#         for t in word_tokenize(doc):
#             lemma.append(self.wnl.lemmatize(t))
        return filter(lambda token: len(token) != 1, lemmas)

def labeledList_to_dict(clustered_txt_, sentences_): 
    '''Display each cluster label followed by data points belonging to that cluster. '''
    clusterDict = collections.defaultdict(list)
    for idx, label in enumerate(clustered_txt_):
        clusterDict[label].append(sentences_[idx])
    return clusterDict

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
        
#     def averaged_sils_by_cluster(self, num_clusters_):
#         '''Aggregate the silhouette scores for samples belonging to cluster i'''    
#         avg_clus_silhouette = [None] * num_clusters_ # initialize Series to hold avg Sils          
#         for clusID in xrange(num_clusters_):
#             ith_cluster = self.select_cluster(clusID)
#             avg_clus_silhouette[clusID] = self.avg_list(ith_cluster)
#         return avg_clus_silhouette
    
    def averaged_sils_by_cluster(self, num_clusters_):
        '''Aggregate the silhouette scores for samples belonging to cluster i'''    
#         avg_clus_silhouette = [None] * num_clusters_ # initialize Series to hold avg Sils
        avg_clus_silhouette = pd.Series(index = range(num_clusters_)) # initialize Series to hold avg Sils          

        for clusID in xrange(num_clusters_):
            ith_cluster = self.select_cluster(clusID)
            avg_clus_silhouette[clusID] = self.avg_list(ith_cluster)
        return avg_clus_silhouette.sort_values(ascending = False)    


# In[3]:

#avg_clus_silhouette = [None] * 3 # initialize Series to hold avg Sils          
#print avg_clus_silhouette 
# avg_clus_silhouette = pd.Series(index=range(4)) # initialize Series to hold avg Sils          
# print avg_clus_silhouette
#Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)


# In[4]:

################################################################
# 1a. GET LINE BY LINE TEXT INPUT OF BKMK TITLES
################################################################
# with open('C:/Users/John/OneDrive/bestk01/sampleBks.txt') as f:
#     sentences = f.read().splitlines()
with open('C:/Users/jjung/Documents/bestk01/sampleBks.txt') as f:
    sentences = f.read().splitlines()
len(sentences)    

################################################################
# 1b. CONVERT RAW TEXT INTO DOC-TERM-MATRIX
################################################################   
# Strip away numbers, punctuations, unicodes, and hexadecimals from string, before lemmatizing/tokenizing
only_letters_spaces = [re.sub('[^A-Za-z\s]+', ' ', sentence) for sentence in sentences]

# produce term-document matrix after 1-gram tokenizing, lemmatizing(stemming), and filtering English stop-words
Vectorizer = TfidfVectorizer(tokenizer = LemmaTokenizer(), stop_words = 'english', strip_accents = 'unicode')   

# term_doc_matrix is a Sparse Matrix object
doc_term_matrix = Vectorizer.fit_transform(only_letters_spaces)

terms = Vectorizer.get_feature_names()
#len(terms)
# 363 terms


# In[42]:

################################################################
# 2. First, Parameter Grid Search: to discover best K (and corresponding avg silhouette) 
# for each parameter configuration on a devised Parameter Grid. 
# Second, store entire silhouette array for each parameter config into a dict.
################################################################
# Create an iterable of grid points for parameter combinations I want to try.
# distance = [euclidean_distance, cosine_distance]
# starts = [5, 10]

distance = [cosine_distance]
starts = [5]
parameter_grid = product(distance, starts)

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


# In[43]:

################################################################
# 3. Select the top 10 non-one (singleton) clusters by highest silhouette score into a dict keyed by parameters
# and valued by Series holding silhouettes, with index of these Series the clusterID's corresponding to each silhouette.
################################################################
f = lambda ser: ser[ser != 1][:min(10, len(ser))]
top_clusters = {k: f(v) for k, v in cluster_sils_by_param.iteritems()}

for k, v in top_clusters.iteritems():
    print k
    print v
    print
# best is (<function cosine_distance at 0x00000000087363C8>, 5)


# In[67]:

################################################################
# 4. Display what the clustered sentences look like by first fitting Kmeans to term-doc matrix using best_k, 
# then sorting by sil score, finally selecting the top 10.
################################################################
best_k = find_best_k(doc_term_matrix.toarray()
                                 , K_CLUSTERS = 20
                                 , dist_measure = cosine_distance
                                 , repeats = 5)

# I. Create clustered list corresponding to inputs.
Clusterer = KMeansClusterer(num_means = best_k
                            , distance = cosine_distance
                            , repeats = 5
                            , avoid_empty_clusters = True) # Instantiate a KMeans object.  
cluster_labels = np.array(Clusterer.cluster(doc_term_matrix.toarray(), assign_clusters=True)) 

# II. Compute top 10 best sil clusters; store in 'top_clusters'. 
sil = Silhouette(doc_term_matrix, cluster_labels)
top_clusters = f(sil.averaged_sils_by_cluster(Clusterer.num_clusters()))

# III. Join to the original input--'sentences'.
clustering = collections.defaultdict(list)
for idx, label in enumerate(cluster_labels):
    clustering[label].append(sentences[idx])

# Display what phrases belong to the top 10 clusters as found by avg silhouette.    
{k: clustering[k] for k in top_clusters.index}


# In[ ]:

# IDEA 1 Should I look at the std of silhouettes in a cluster? A very low silhouette in an otherwise high avg silhouette 
# cluster could be cause for concern.
# IDEA 2 hand label bkmks into folders, so I can check Gini Impurity.
# IDEA 3 lsa = TruncatedSVD(n_components=60)
# X_lsa = lsa.fit_transform(doc_term_matrix)
# Clusterer = KMeans(n_clusters = best_k)
# Clusterer.fit(X_lsa)

