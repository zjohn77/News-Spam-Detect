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