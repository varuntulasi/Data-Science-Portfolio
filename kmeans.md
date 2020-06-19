## Kmeans clustering

Kmeans clustering is a popular and one of the simplest unsupervised learning algorithms and is used when you have unlabeled data (meaning data without groups or categoies).
The algorithm discovers groups in the data, with the variable <em>K</em> representing the number of groups it discovers. It iteratively assigns each data point, based on the features provided, to one of the <em>K</em> groups wherein the data points are clustered using feature similarity.
In other words, the algorithm groups similar data points together and finds underlying patters by looking for a fixed number (<em>K</em>) of clusters in the data.
There are two results from this algorithm: 

1. Labels for the data wherein each data point is allotted a single cluster.

2. To label new data, centroids of <em>K</em> clusters are found. A centroid is a real of imaginary point representing the center of the cluster and each centroid in a cluster is a collection of feature values which describe the resulting groups. By inspecting the centroid feature weights, one can qualitatively explain what kind of label of group each cluster represents.

