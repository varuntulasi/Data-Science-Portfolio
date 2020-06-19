## K-means clustering

Kmeans clustering is a popular and one of the simplest unsupervised learning algorithms and is used when you have unlabeled data (meaning data without groups or categoies).
The algorithm discovers groups in the data, with the variable <em>K</em> representing the number of groups it discovers. It iteratively assigns each data point, based on the features provided, to one of the <em>K</em> groups wherein the data points are clustered using feature similarity.
In other words, the algorithm groups similar data points together and finds underlying patters by looking for a fixed number (<em>K</em>) of clusters in the data.
There are two results from this algorithm: 

1. Labels for the data wherein each data point is allotted a single cluster.

2. To label new data, centroids of <em>K</em> clusters are found. A centroid is a real of imaginary point representing the center of the cluster and each centroid in a cluster is a collection of feature values which describe the resulting groups. By inspecting the centroid feature weights, one can qualitatively explain what kind of label or group each cluster belongs to.

The way K-means algorithm works is it first starts with a randomly selected centroids which it uses as the starting point for every cluster and then it performs repetitive calculations and optimizes the positions of the centroids. It halts this process either when the centroids have stabilized and there is not change in its values or if the defined number of iterations has been reached.

In our project, after preprocessing the feedbacks we tried various implementations of K-means. We performed K-means using Term frequency–inverse document frequency (Tfidf), pretrained Word2Vec, Glove and Fasttext. 

Using Tfidf we instantiate TfidfVectorizer() vectorizer and fit_transform the feedbacks. To determine the (<em>K</em>) number of clusters to pick, we created an elbow_curve function to calculate the elbow curve using inertia, distortion score and silhouette score. We could not find any particular paper that mentioned which metric is more accurate than the other in a given situation so we decided get the (<em>K</em>) value from the three metrics and print out the corresponding clusters to evaluate their performance. In order to print the clusters, we created cluster_samples function that, given the (<em>K</em>) value, feedbacks and the number of samples (n), would run the K-means with (<em>K</em>) clusters and print n samples from each cluster.

When it came to using pretrained Word2Vec, glove and Fasttext, we used gensim to load the model and created sentence_vectorizer function which, given a model and a feedback sentence, would sum the embeddings for each word in the feedback sentence and divide it by the count of the feedback sentence. Then we reduced the dimentionality to 20 using Principal component analysis (PCA) making sure that the explained variance ratio after the reduction was above 90%. 
