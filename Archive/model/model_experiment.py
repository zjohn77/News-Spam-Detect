##### MODEL
def stopCondition(metric_cur, metric_prev, k_cur, k_prev, nrow):
    reached_elbow = (metric_cur - metric_prev) / (k_cur - k_prev) < metric_cur / k_cur
    return reached_elbow or (k_cur >= nrow)

def best_model(X, clusterQual, stopCondition):
    while True:
        kmeans_fit = KMeans(n_clusters = k).fit(X)        
        
        # roll forward pointers
        metric_prev, metric_cur = metric_cur, clusterQual(X, labels = kmeans_fit.labels_)
        k_prev, k_cur = k_cur, k_cur*2
        
        if stopCondition(metric_cur, metric_prev, k_cur, k_prev, nrow):
            break