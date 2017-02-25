from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score
import matplotlib.pyplot as plt

##### MODEL
def __fit_seq_clusters(X, bound):
    '''Fit k-means once for every k between 2 and some bound; return {k: clustering} pairs as a dict.'''
    return {k: KMeans(k).fit(X).labels_ for k in range(2, bound)}    

def __calc_metrics(k_cluster_pairs, X):
    '''For every clustering, compute the silhouette and the calinski score.'''
    silhou, calins = {}, {}
    for key, value in k_cluster_pairs.items():   
        silhou[key] = silhouette_score(X, labels=value, metric='cosine')
        calins[key] = calinski_harabaz_score(X, labels=value)   
    return (silhou, calins)           
    # return {silhouette: silhou,
    #         calinski: calins}    

def __stack_plot(x, y1, y2, ticks, x_label='no. of clusters', y1_label='Silhouette Score', y2_label='Calinski Score'): 
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

def sensitiv(X):
    ### 1. Loop over k, calling sklearn's KMeans at each iteration
    ### save the predicted cluster labels as a dict keyed by k.     
    k_metric_pairs = __calc_metrics(__fit_seq_clusters(X, len(X)), X)
    __stack_plot(x = list(k_metric_pairs[0].keys()), 
                y1 = list(k_metric_pairs[0].values()), 
                y2 = list(k_metric_pairs[1].values()),                 
                ticks = range(2, len(X)))

    

### 3.  
# x = list(silhou1.keys())
# x_label = 'no. of clusters'
# y1, y2 = list(silhou1.values()), list(calinski1.values())
# y1_label, y2_label = 'Silhouette Score', 'Calinski Score'
# ticks = range(2, len(X1))
# stack_plot(x, x_label, y1='Silhouette Score', y2, y1_label, y2_label, ticks)
# stack_plot(x, x_label, y1, y2, y1_label, y2_label, ticks)