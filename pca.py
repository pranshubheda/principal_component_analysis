import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans

if __name__ == "__main__":
    #read the data
    data = pd.read_csv("HW_PCA_SHOPPING_CART_v892.csv")
    data = data.values
    #removing id
    data = np.delete(data, 0, 1)
    #calculate the covariance matrix
    covariance_matrix = np.cov(data.T)
    #calculate the eigen values and vectors for covariance matrix
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    #create tuples of eigen_value, eigen_vector pair
    eigen_tuples = list(zip(np.abs(eigen_values), eigen_vectors))
    #sort the eigen vectors using eigen values in descending order
    eigen_tuples.sort(key = lambda x : x[0], reverse=True)
    #normalized eigen values representative of percentage variance of original data
    normalized_eigen_values = eigen_values/ np.sum(eigen_values)
    #calculate cumulative eigen values sum
    cumulative_eigen_values_sum = []
    cumulative_eigen_values_sum.append(normalized_eigen_values[0])
    for i in range(1, len(normalized_eigen_values)):
        cumulative_eigen_values_sum.append(normalized_eigen_values[i]+cumulative_eigen_values_sum[i-1]) 
    plt.plot(np.arange(len(cumulative_eigen_values_sum)), cumulative_eigen_values_sum, color="blue", marker="o", linestyle="dashed", linewidth=0.5)
    plt.title("Cumulative Variance vs Number of Eigenvalues used")
    plt.xlabel("Number of Eigenvalues used")
    plt.ylabel("Cumulative Amount of Variance Captured")
    plt.show()
    # displaying the first five eigen value - eigen vectors pair
    for i in range(5):
        eigen_tuple = eigen_tuples[i]
        print("Eigenvalue: {}\nEigenvector: {}\n".format(np.round(eigen_tuple[0], 1), np.round(eigen_tuple[1], 1)))
    #projecting data on the first 3 principal components
    principal_components_one_till_three = [np.round(eigen_tuples[i][1], 1) for i in range(3)]
    principal_components_one_till_three = np.array(principal_components_one_till_three)
    projected_data = data.dot(principal_components_one_till_three.T)
    #visualize the projected data points in 3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(projected_data[:,0], projected_data[:,1], projected_data[:,2])
    plt.title("Projected Data")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()
    #cluster the projected data points
    kmeans = KMeans(n_clusters=6, random_state=13)
    kmeans.fit(projected_data)
    y_kmeans = kmeans.predict(projected_data)
    #plot clusters in 3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(projected_data[:,0], projected_data[:,1], projected_data[:,2], c=y_kmeans, cmap='viridis')
    #plot cluster centers
    centers = kmeans.cluster_centers_
    ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.5)
    plt.title("Clustered Data with K=6")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()