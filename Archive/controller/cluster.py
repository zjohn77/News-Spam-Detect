from sklearn.metrics import silhouette_score, silhouette_samples
import string
import re
import collections
from pprint import pprint
from nltk.tokenize import word_tokenize
from nltk.cluster import KMeansClusterer, cosine_distance

################################################################
# 4. Display what the clustered sentences look like by first fitting Kmeans to term-doc matrix using best_k, 
# then sorting by sil score, finally selecting the top 10.
################################################################
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