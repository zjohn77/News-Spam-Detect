def labeledList_to_dict(clustered_txt_, sentences_): 
    '''Display each cluster label followed by data points belonging to that cluster. '''
    clusterDict = collections.defaultdict(list)
    for idx, label in enumerate(clustered_txt_):
        clusterDict[label].append(sentences_[idx])
    return clusterDict