# def kBounds(X):
#   return range(2, sqrt(X.shape[0])) 

### 3. 
def labeledList_to_dict(clustered, docs): 
    '''Display each cluster label followed by data points belonging to that cluster. '''
    result_dict = defaultdict(list)
    for i, label in enumerate(clustered):
        result_dict[label].append(docs[i])
    return result_dict

# pprint(labeledList_to_dict(model_found, docs))

# # III. Join to the original input--'sentences'.
# # Display what phrases belong to the top 10 clusters as found by avg silhouette.    
# {k: clustering[k] for k in top_clusters.index}

### Bootstrap this algorithm multiple times using a smaller sample; 
### plot the resulting (metric, k) to see the impact of randomness/outliers