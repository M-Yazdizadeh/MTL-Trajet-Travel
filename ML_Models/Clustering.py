# Import necessary libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Data
# Load the dataset into a pandas DataFrame
data = pd.read_csv('first_100000_rows.csv')  # Replace with your actual dataset path

# Step 2: Drop Unnecessary Features
# Remove features that are redundant or not useful for clustering
# - 'h_accuracy', 'v_accuracy': GPS measurement accuracy, not useful for clustering.
# - 'geometry': Redundant as latitude and longitude already represent spatial data.
# - 'id_trip': An identifier that does not provide analytical value.
# - 'timestamp', 'starttime', 'endtime': Drop these unless temporal analysis is required.
data = data.drop(columns=['h_accuracy', 'v_accuracy', 'geometry', 'id_trip', 'timestamp', 'starttime', 'endtime'])

# Step 3: Handle Missing Values
# Separate numeric and categorical columns for proper preprocessing
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns  # Numeric features
categorical_columns = data.select_dtypes(include=['object']).columns       # Categorical features

# Impute missing values for numeric columns using the mean
# This replaces NaN values in numeric columns with the mean of each column
numeric_imputer = SimpleImputer(strategy='mean')
numeric_data = pd.DataFrame(
    numeric_imputer.fit_transform(data[numeric_columns]),
    columns=numeric_columns
)

# Impute missing values for categorical columns using the most frequent value
# This replaces NaN values in categorical columns with the most common value in each column
categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_data = pd.DataFrame(
    categorical_imputer.fit_transform(data[categorical_columns]),
    columns=categorical_columns
)

# Step 4: Reduce Cardinality of Categorical Features (Optional)
# For high-cardinality categorical features, reduce the number of unique categories
# by grouping less frequent categories into an "Other" category
def limit_categories(df, column, top_n=10):
    """
    Function to limit the number of unique categories in a categorical column.
    Keeps the top_n most frequent categories and groups the rest into 'Other'.
    """
    top_categories = df[column].value_counts().nlargest(top_n).index  # Get the top_n categories
    df[column] = df[column].where(df[column].isin(top_categories), 'Other')  # Replace less frequent categories
    return df

# Apply the category reduction function to each categorical column
for col in categorical_columns:
    categorical_data = limit_categories(categorical_data, col)

# Step 5: Encode Categorical Data
# Transform categorical data into numerical format using OneHotEncoding
# Sparse format is used to save memory when dealing with large datasets
encoder = OneHotEncoder(drop='first', sparse_output=True)  # Drop first to avoid multicollinearity
encoded_data_sparse = encoder.fit_transform(categorical_data)

# Step 6: Combine Numeric and Categorical Data
# Combine the processed numeric and categorical data into a single dataset
# Use sparse matrix format for efficient memory usage
numeric_data_sparse = csr_matrix(numeric_data)  # Convert numeric data to sparse format
combined_sparse_data = hstack([numeric_data_sparse, encoded_data_sparse])  # Horizontally stack numeric and encoded data

# Step 7: Scale the Data
# Standardize the data to ensure all features contribute equally to clustering
# Use with_mean=False because sparse matrices do not support centering
scaler = StandardScaler(with_mean=False)
scaled_data = scaler.fit_transform(combined_sparse_data)

# Step 8: Dimensionality Reduction with PCA
# Apply Principal Component Analysis (PCA) to reduce the dataset to 2 dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Step 9: Examine PCA Loadings
# Display the contribution of each feature to the two principal components
loadings = pd.DataFrame(
    pca.components_,
    columns=list(numeric_columns) + encoder.get_feature_names_out().tolist(),
    index=['PCA1', 'PCA2']
)

print("\nPCA Loadings:")
print(loadings)

# Identify top contributors to PCA1 and PCA2
top_contributors_pca1 = loadings.loc['PCA1', :].sort_values(ascending=False).head(10)
top_contributors_pca2 = loadings.loc['PCA2', :].sort_values(ascending=False).head(10)

print("\nTop contributors to PCA1 (Feature Names and Weights):")
for feature, weight in top_contributors_pca1.items():
    print(f"Feature: {feature}, Weight: {weight:.4f}")

print("\nTop contributors to PCA2 (Feature Names and Weights):")
for feature, weight in top_contributors_pca2.items():
    print(f"Feature: {feature}, Weight: {weight:.4f}")

# Step 10: Apply K-Means Clustering
# Perform K-Means clustering on the PCA-reduced data
kmeans = KMeans(n_clusters=3, random_state=42)  # Set number of clusters to 3
kmeans_labels = kmeans.fit_predict(reduced_data)  # Fit and predict cluster labels

# Step 11: Visualize K-Means Clusters
# Create a scatter plot to visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=kmeans_labels, palette='viridis')
plt.title('K-Means Clustering Visualization (PCA Reduced)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.savefig('kmeans_clusters.png')  # Save the figure as a PNG file
plt.show()

# Step 12: Apply DBSCAN Clustering (Optional)
# Perform DBSCAN clustering to identify density-based clusters
dbscan = DBSCAN(eps=0.3, min_samples=15)  # Set DBSCAN parameters
dbscan_labels = dbscan.fit_predict(reduced_data)  # Fit and predict cluster labels

# Visualize DBSCAN Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=dbscan_labels, palette='viridis')
plt.title('DBSCAN Clustering Visualization (PCA Reduced)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.savefig('dbscan_clusters.png')  # Save the figure as a PNG file
plt.show()
