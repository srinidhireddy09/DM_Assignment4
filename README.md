# DM_Assignment4
This project involves clustering a 4-class dog image dataset using various unsupervised learning techniques. The primary steps include:

Feature Extraction from the last convolutional layer of a pre-trained ResNet18 model.
Dimensionality Reduction to reduce the feature space to 2D for visualization and clustering.
Clustering using several algorithms including K-means, Spectral Clustering, DBSCAN, Agglomerative Clustering, and Bisecting K-means.
Clustering Evaluation using the Fowlkes-Mallows index and the Silhouette Coefficient to evaluate the quality of the clusters.
## Dataset
This dataset consists of images of dogs belonging to four distinct classes. Each image is resized to 224x224 pixels to match the input size of the ResNet18 model.
## Dependencies
This project requires the following libraries:

Python 3.x
PyTorch (for feature extraction with ResNet18)
Scikit-learn (for clustering and evaluation)
Matplotlib (for visualization)
NumPy (for numerical operations)
OpenCV or PIL (for image preprocessing)
## steps
1. Feature Extraction
To extract features from the last convolutional layer of the ResNet18 model:

Image Preprocessing:
Resize each image to 224x224 pixels.
Normalize the pixel values to have a mean of 0.485, 0.456, 0.406 and a standard deviation of 0.229, 0.224, 0.225 (as used in ImageNet).
Extract Features using ResNet18:
Load the pre-trained ResNet18 model from PyTorch's torchvision.models.
Register a forward hook to capture the output of the last convolutional layer.
Pass each image through the model to extract feature vectors from the last convolutional layer.
2. Dimensionality Reduction
Perform PCA (Principal Component Analysis) or t-SNE (t-distributed Stochastic Neighbor Embedding) to reduce the dimensionality of the feature vectors to 2D.
3. Clustering
Apply the following clustering algorithms to the 2D feature vectors:

K-means with init='random' and init='k-means++'.
Bisecting K-means.
Spectral Clustering.
DBSCAN: Choose appropriate eps and min_samples values to get 4 clusters.
Agglomerative Clustering with different linkages: 'single', 'complete', 'average', and 'ward'.
4. Results and Ranking
For each clustering method, compute the Fowlkes-Mallows index and Silhouette Coefficient. Rank the methods based on their performance (higher values are better).

