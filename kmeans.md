## K-means clustering

Kmeans clustering is a popular and one of the simplest unsupervised learning algorithms and is used when you have unlabeled data (meaning data without groups or categoies).
The algorithm discovers groups in the data, with the variable <em>K</em> representing the number of groups it discovers. It iteratively assigns each data point, based on the features provided, to one of the <em>K</em> groups wherein the data points are clustered using feature similarity.
In other words, the algorithm groups similar data points together and finds underlying patters by looking for a fixed number (<em>K</em>) of clusters in the data.
There are two results from this algorithm: 

1. Labels for the data wherein each data point is allotted a single cluster.

2. To label new data, centroids of <em>K</em> clusters are found. A centroid is a real of imaginary point representing the center of the cluster and each centroid in a cluster is a collection of feature values which describe the resulting groups. By inspecting the centroid feature weights, one can qualitatively explain what kind of label or group each cluster belongs to.

The way K-means algorithm works is it first starts with a randomly selected centroids which it uses as the starting point for every cluster and then it performs repetitive calculations and optimizes the positions of the centroids. It halts this process either when the centroids have stabilized and there is no change in its values or if the defined number of iterations has been reached.

In our project, after preprocessing the feedbacks we tried various implementations of K-means. We performed K-means using Term frequency–inverse document frequency (Tfidf), pretrained Word2Vec, Glove and Fasttext. 

Using Tfidf we instantiated TfidfVectorizer() vectorizer and fit_transform the feedbacks. To determine the (<em>K</em>) number of clusters to pick, we created an elbow_curve function to calculate the elbow curve using inertia, distortion score and silhouette score. We could not find any particular paper that mentioned which metric is more accurate than the other in a given situation so we decided get the (<em>K</em>) value from the three metrics and printed out the corresponding clusters to evaluate their performance. In order to print the clusters, we created cluster_samples function that, given the (<em>K</em>) value, feedback sentences and the number of samples (n), would run the K-means with (<em>K</em>) clusters and randomly print n sample feedbacks from each cluster.

When it came to using pretrained Word2Vec, glove and Fasttext, we used gensim to load the model and created sentence_vectorizer function which, given a model and a feedback sentence, would sum the embeddings for each word in the feedback sentence and divide it by the count of the feedback sentence. Then using the elbow_curve and cluster_samples functions, we got the (<em>K</em>) value and printed 15 samples from each cluster for each each of the models.

To improve the performance and usefulness of clusters we tried a couple of things. We reduced the dimentionality to 20 before performing K-means using Principal component analysis (PCA) making sure that the explained variance ratio after the reduction was above 90%. Then we tried doing normalization before K-means by changing the sentence_vectorizer function because K-means is unable to handle noisy data and missing values and normalization sometimes helps to eliminate redundant data and improve the efficiency of clustering algorithms. We also tried tuning some of K-means hyperparemeters to check if it improved the quality of the clusters. And finally, we also used the chat logs provided by Unbounce and after filtering and preprocessing it, tried two things with it. First we added it to the original feedbacks and performed K-means using pretrained Word2Vec and second we trained word2vec embeddings on the chat log data and then used those to cluster the feedbacks.


## Evaluation

Evaluation of the models in this project turned out to be particularly challenging. The lack of labels on an unsupervised learning model's training data made evaluation problematic because there was nothing to which the model's results could be meaningfully compared. During the first few weeks of this project, we did a literature review to understand how evaluated unsupervised models. There were our findings:


#### K-means

To decide the right number of clusters or K we can plot a graph known as the **elbow curve** wherein the x-axis represents the number of clusters and the y-axis represents an evaluation metric such as inertia. It is based on the sum of squared distance (SSE) between data points and their clusters' centroids. The spot where SSE starts to flatten out and forms an elbow is the K we pick. Another option would be to perform **silhouette analysis** `sklearn.metrics.silhouette_score` wherin the degree of separation between clusters is determined. In this metric, we first compute the average distance from all data points in the same cluster and then in the closest cluster and compute the coefficient. The coefficient takes the values between -1 and 1 and a coefficient closer to 1 indicates a good cluster.

#### DBScan

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a popular clustering method used to separate clusters of high density from clusters of low density. To measure its performance, silhouette score `sklearn.metrics.silhouette_score` can be used wherein the mean nearest-cluster distance and the intra-cluster distance between points is utilized. It ranges from -1 to 1 with 1 meaning the best score. A strong silhouette score is achieved when a cluster with a lot of data points are very close one another and far away from the next nearest cluster. Davies-Bouldin index `sklearn.metrics.davies_bouldin_score` where a value close to 0 indicates a good cluster and Calinski-Harabasz index `sklearn.metrics.calinski_harabasz_score` where a higher score relates to a model with better defined cluser can also be used to judge the performance of DBScan clustering.

#### Expectation–Maximization (EM) Clustering using Gaussian Mixture Models

The Gaussian mixture models can be used the same way as k-means to cluster unlabeled data. However there are two differences. The first being that Kmeans does not account for variance and secondly unlike kmeans, gaussian mixed models give probabilities that a given data point belongs to each of the possible clusters instead of just telling us which data point belongs to which cluster. To evalute the quality of clustering, this [paper](https://ieeexplore.ieee.org/abstract/document/8605459) uses silhouette index. The Jensen-Shannon (JS) metric can also be used to evalute the performance by randomly splitting the original dataset into two datasets and checking how similar the two GMMs trained datasets are for each configuration. The code to calculate the Jensen-Shannon metric can be found [here](https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms). And finally, this [paper](https://www.isca-speech.org/archive/archive_papers/icslp_2000/i00_3714.pdf) uses Bayesian information criterion (BIC) in `sklearn.mixture.GaussianMixture` to evaluate the accuracy of the clustering.


### Topic Modeling Evaluation - LDA

Evaluating probabilistic topic models such as LDA is challenging due to its unsupervised training process and as there is no gold standard list of topics to compare against our corpus. Topic coherence measure is a good way to, based on their human-interpretability, compare different topic models. In it a score is given for single topic by measuring the degree of semantic similarity between high scoring words in the topic and it helps to distinguish topics that are artifacts of statistical inference and topics that are semantically interpretable. The two measures of Topic coherence are Intrinsic and Extrinsic. In the [paper](https://www.aclweb.org/anthology/D12-1087.pdf), UMass is used as the representation for intrinsic measure. It requires an ordered words set in which a word is only compared to preceding and succeeding words respectively. It uses as pairwise score function which is the empirical log-probability with smoothing. For extrinsic measure, UCI is used which pairs every single word with every other word. It uses the pointwise mutual information (PMI).
