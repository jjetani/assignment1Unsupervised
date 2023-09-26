#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Question 1:


# In[2]:


#import necessary libraries:
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


# In[3]:


mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print(mnist)


# In[4]:


# Display each digit from the MNIST dataset

# Convert data and labels to unsigned 8-bit integers
data = mnist.data.astype(np.uint8)
labels = mnist.target.astype(np.uint8)

# Create a figure for displaying the digits
plt.figure(figsize=(10, 10))

# Loop through each digit (0-9)
for i in range(10):
    # Find the indices of samples that belong to the current digit
    digit_indices = np.where(labels == i)[0]
    
    # Create a subplot for displaying 10 samples of the current digit
    for j in range(10):
        plt.subplot(10, 10, i * 10 + j + 1)
        
        # Display the image of the digit
        plt.imshow(data[digit_indices[j]].reshape(28, 28), cmap='gray')
        
        # Set the title to indicate the current digit
        plt.title(f"Digit: {i}")
        
        # Turn off axis labels and ticks
        plt.axis('off')
    
    # Show the current set of 10 digit samples
    plt.show()
    
    # Create a new figure for the next digit
    plt.figure(figsize=(10, 10))

# Ensure proper layout and display the entire grid of digits
plt.tight_layout()
plt.show()


# In[5]:


# Number of principal components you want to keep
n_components = 2

# Initialize PCA with the desired number of components
pca = PCA(n_components=n_components)

pca.fit(data)

# Retrieve the 1st and 2nd principal components
first_principal_component = pca.components_[0]
second_principal_component = pca.components_[1]

explained_variance_ratio = pca.explained_variance_ratio_

print(f"Explained Variance Ratio for 1st Principal Component: {explained_variance_ratio[0]}")
print(f"Explained Variance Ratio for 2nd Principal Component: {explained_variance_ratio[1]}")


# In[6]:


# Project the data onto the 1D hyperplane defined by the 1st principal component
projection_1st_component = np.dot(data, first_principal_component)

# Project the data onto the 1D hyperplane defined by the 2nd principal component
projection_2nd_component = np.dot(data, second_principal_component)

# Create a scatter plot for the 1st component projection
plt.figure(figsize=(12, 4))  # Create a figure for the plots
plt.subplot(1, 2, 1)  # Create a subplot for the 1st component projection
plt.scatter(projection_1st_component, np.zeros_like(projection_1st_component), c=labels, cmap='viridis', marker='o')  # Scatter plot
plt.title("Projection onto 1st Principal Component")  
plt.xlabel("Projection Value")  # Label the x-axis
plt.yticks([])  # Remove y-axis ticks

# Create a scatter plot for the 2nd component projection
plt.subplot(1, 2, 2)  # Create a subplot for the 2nd component projection
plt.scatter(projection_2nd_component, np.zeros_like(projection_2nd_component), c=labels, cmap='viridis', marker='o')  # Scatter plot
plt.title("Projection onto 2nd Principal Component")  
plt.xlabel("Projection Value")  # Label the x-axis
plt.yticks([])  # Remove y-axis ticks

plt.tight_layout()  # Ensure proper layout
plt.show()  # Display the plots


# In[7]:


# Number of desired dimensions
n_components = 154

# Initialize Incremental PCA with the desired number of components
ipca = IncrementalPCA(n_components=n_components)

# Fit IPCA to the MNIST data in batches
batch_size = 2000
for i in range(0, data.shape[0], batch_size):
    batch = data[i:i+batch_size]
    ipca.partial_fit(batch)

# Transform the entire dataset to the reduced dimensionality
mnist_reduced = ipca.transform(data)


# In[8]:


# Choose a random subset of digits for visualization
sample_indices = np.random.choice(mnist_reduced.shape[0], 10, replace=False)

plt.figure(figsize=(12, 5))

# Display original digits
plt.subplot(1, 2, 1)
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 10, i + 1)
    plt.imshow(ipca.inverse_transform(mnist_reduced[idx]).reshape(28, 28), cmap='gray')
    plt.title(f'Original')
    plt.axis('off')

# Display compressed digits
plt.subplot(1, 2, 2)
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 10, i + 11)
    plt.imshow(ipca.inverse_transform(mnist_reduced[idx]).reshape(28, 28), cmap='gray')
    plt.title(f'Compressed')
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[9]:


#Question 2:


# In[10]:


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# In[11]:


# Generate the Swiss Roll dataset:
n_samples = 1000
X, color = make_swiss_roll(n_samples, noise=0.2, random_state=42)


# In[12]:


# Generate the Swiss Roll dataset
n_samples = 1000
noise = 0.2
t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples, 1))
X = np.concatenate([t * np.cos(t), 10 * np.random.rand(n_samples, 1), t * np.sin(t)], axis=1)
color = t.reshape(-1)

# Create a 3D scatter plot for the Swiss Roll dataset
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color mapping
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

# Set the title of the plot
ax.set_title("Swiss Roll Dataset")

# Show the plot
plt.show()


# In[13]:


# Define the kPCA models with different kernels
kpca_linear = KernelPCA(kernel="linear", n_components=2)
kpca_rbf = KernelPCA(kernel="rbf", gamma=0.04, n_components=2)
kpca_sigmoid = KernelPCA(kernel="sigmoid", gamma=0.001, n_components=2)

# Fit and transform the data using each kPCA model
X_linear = kpca_linear.fit_transform(X)
X_rbf = kpca_rbf.fit_transform(X)
X_sigmoid = kpca_sigmoid.fit_transform(X)

# Plot the results
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(X_linear[:, 0], X_linear[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("kPCA with Linear Kernel")

plt.subplot(132)
plt.scatter(X_rbf[:, 0], X_rbf[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("kPCA with RBF Kernel")

plt.subplot(133)
plt.scatter(X_sigmoid[:, 0], X_sigmoid[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("kPCA with Sigmoid Kernel")

plt.tight_layout()
plt.show()


# In[14]:


# Plot the result of GridSearchCV
scores = grid_search.cv_results_["mean_test_score"]
gammas = [params["kpca__gamma"] for params in grid_search.cv_results_["params"]]
plt.figure(figsize=(10, 6))
plt.scatter(gammas, scores, c=scores, cmap=plt.cm.viridis)
plt.colorbar(label="Mean Test Score")
plt.xlabel("Gamma")
plt.ylabel("Mean Test Score")
plt.title("GridSearchCV Results")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




