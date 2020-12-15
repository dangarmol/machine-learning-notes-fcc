# Clustering
The first form of unsupervised learning we are going to see. It’s only useful for a specific set of problems, usually when you have a bunch of data but you don’t have any labels or output information.

### K-Means algorithm
Clustering simply groups data in clusters and gives you the location of those groups.
We need to understand the concept of <ins>centroid</ins>. Basically, centroids start being random points in the data, we need to decide how many groups there will be, and for each one we will have a centroid. First we randomly place the centroid, we calculate the centroid each actual data point belongs to (the closest). Once all data points are classified, we can calculate the “center of mass”, which is somewhat an average of all data points of that group. Then we repeat this process several times, keeping in mind that different data points can move from one group to another in the process. We keep doing this until the data points don’t really change group anymore.
The problem here is that you need to specify the amount of clusters you want, although there are some algorithms that help with this.
