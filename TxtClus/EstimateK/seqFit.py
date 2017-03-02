from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import matplotlib.pyplot as plt

##### MODEL
def __fit_seq_clusters(X, bound):
    '''Fit k-means once for every k between 2 and some bound; return {k: clustering} pairs as a dict.'''
    return {k: KMeans(k).fit(X).labels_ for k in range(2, bound)}    

def __calc_metrics(k_cluster_pairs, X):
    '''For every clustering, compute the silhouette and the calinski score.'''
    metrics = DataFrame()
    for key, value in k_cluster_pairs.items():   
        metrics[key] = [silhouette_score(X, labels=value, metric='cosine'), 
                        calinski_harabaz_score(X, labels=value)] 
    metrics.index = ['silhouette', 'calinski']                      
    return metrics 

def sensitiv(X):
    '''Controller that loops over k, calling sklearn's KMeans at each iteration
     save the predicted cluster labels as a dict keyed by k.'''     
    k_metric_pairs = __calc_metrics(__fit_seq_clusters(X, X.shape[0]), X)
    return k_metric_pairs

def stack_plot(x, y1, y2, ticks, x_label='no. of clusters', y1_label='Silhouette Score', y2_label='Calinski Score'): 
    '''Visualize how each metric varies as a function of K (no. of clusters).'''
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False) # create 2x1 subplots
    ax1.plot(x, y1)
    ax1.set_xticks(ticks)
    ax1.set_ylabel(y1_label)

    ax2.plot(x, y2, c="green")
    ax2.set_xticks(ticks)
    ax2.set_ylabel(y2_label)
    ax2.set_xlabel(x_label)
    plt.show()    