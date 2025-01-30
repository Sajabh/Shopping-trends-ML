import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

################### Compute Basic Statistics
# Load your dataset
df = pd.read_csv('shopping_trends.csv')

# 1. Summary of the dataset
print("Dataset Overview:")
print(df.info())  # General info about the dataset

# 2. Numerical Columns Statistics
numerical_columns = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases', 'Frequency of Purchases']
print("\nNumerical Columns Statistics:")
print(df[numerical_columns].describe())  # Mean, Min, Max, Std, etc.

# 3. Frequency Counts for Categorical Columns
categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Payment Method']
print("\nCategorical Columns Frequency Counts:")
for col in categorical_columns:
    print(f"\nFrequency counts for {col}:")
    print(df[col].value_counts())

################### Visualization ###################
# Increase plot size
plt.rcParams["figure.figsize"] = (10, 6)

# 1. Histograms for Numerical Data
numerical_columns = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases', 'Frequency of Purchases']
for col in numerical_columns:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=20, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 2. Boxplots for Numerical Data (Outlier Detection)
for col in numerical_columns:
    plt.figure()
    sns.boxplot(x=df[col], color='orange')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()

# 3. Bar Charts for Categorical Data
categorical_columns = ['Gender', 'Category', 'Location', 'Payment Method']
for col in categorical_columns:
    plt.figure()
    df[col].value_counts().plot(kind='bar', color='green')
    plt.title(f'Frequency of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# 4. Pie Chart for Subscription Status
if 'Subscription Status' in df.columns:
    plt.figure()
    df['Subscription Status'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['gold', 'skyblue'], startangle=90)
    plt.title('Subscription Status Distribution')
    plt.ylabel('')
    plt.show()

################### Transform the Dataset to Numerical Values ###################

# 1. Identify categorical columns
categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Payment Method']

# 2. Label Encoding (Example: Gender)
label_encoder = LabelEncoder()
df['Gender_Encoded'] = label_encoder.fit_transform(df['Gender'])
print("\nGender Label Encoding:")
print(df[['Gender', 'Gender_Encoded']].head())

# 3. One-Hot Encoding (Example: Category)
df_one_hot = pd.get_dummies(df, columns=['Category'], prefix='Category')
print("\nOne-Hot Encoded Columns for Category:")
print(df_one_hot.head())

# 4. Drop original categorical columns if they exist
df_transformed = df_one_hot.drop(categorical_columns, axis=1, errors='ignore')
print("\nDataset after dropping original categorical columns:")
print(df_transformed.head())

# Encode 'Frequency of Purchases' as ordinal (Weekly > Fortnightly > Quarterly)
frequency_mapping = {'Weekly': 3, 'Fortnightly': 2, 'Quarterly': 1}
df['Frequency of Purchases'] = df['Frequency of Purchases'].map(frequency_mapping)

################### missing values in the dataset ###################
missing_data = df.isnull().sum()
print("Missing Values in Each Column:\n", missing_data)

# Total missing values
print("\nTotal Missing Values:", missing_data.sum())

# Handle missing values in numerical columns (using median)
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Handle missing values in categorical columns (using mode)
for col in categorical_columns:
    if col in df.columns:  # Ensure the column exists in the dataset
        df[col] = df[col].fillna(df[col].mode()[0])

# Verify that missing values are handled
print("\nMissing Values After Handling:\n", df.isnull().sum())

################### Transformation/ Normalize data ###################
# Instantiate the scaler
scaler = StandardScaler()

# Scale numerical columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Verify scaling
print("\nScaled Numerical Data (first few rows):")
print(df[numerical_columns].head())

# Example of merging two datasets (if applicable)
# df = pd.merge(df1, df2, on='common_column')

# Drop duplicates
df = df.drop_duplicates()

# Check for duplicates
print("\nTotal Duplicate Rows After Cleaning:", df.duplicated().sum())

################### Correlation matrix ###################
plt.figure(figsize=(10, 6))
sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Reduce dimensions for visualization (optional)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df[numerical_columns])

################### Apply KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Set number of clusters
clusters = kmeans.fit_predict(reduced_data)

# Add cluster labels to the DataFrame
df['Cluster'] = clusters

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette='Set1')
plt.title("Clusters Found in the Dataset")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

################### Set a variance threshold
selector = VarianceThreshold(threshold=0.01)
reduced_features = selector.fit_transform(df[numerical_columns])
print("Reduced Dataset Shape:", reduced_features.shape)

################### Apply PCA
pca = PCA(n_components=5)  # Keep 5 principal components
pca_data = pca.fit_transform(df[numerical_columns])

# Show explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
