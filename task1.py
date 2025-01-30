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
from sklearn.metrics import r2_score, mean_absolute_error, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import cross_val_score, KFold
from deap import base, creator, tools, algorithms
import random
from sklearn.base import ClassifierMixin


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


################### Prepare Data for ML Models ###################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import numpy as np

# Prepare features (X) and target (y)
# For regression tasks, we'll predict Purchase Amount
X_reg = df[['Age', 'Review Rating', 'Previous Purchases', 'Frequency of Purchases']]
y_reg = df['Purchase Amount (USD)']

# For classification tasks, let's create a binary target based on purchase amount median
purchase_median = df['Purchase Amount (USD)'].median()
y_class = (df['Purchase Amount (USD)'] > purchase_median).astype(int)

# Split the data
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_reg, y_class, test_size=0.2, random_state=42)

################### Regression Models ###################
def evaluate_regression(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{model_name} - RMSE: {rmse:.2f}")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_reg_train, y_reg_train)
lr_pred = lr_model.predict(X_reg_test)
evaluate_regression(y_reg_test, lr_pred, "Linear Regression")

# Decision Tree Regression
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_reg_train, y_reg_train)
dt_reg_pred = dt_reg.predict(X_reg_test)
evaluate_regression(y_reg_test, dt_reg_pred, "Decision Tree Regression")

# Random Forest Regression
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_reg_train, y_reg_train)
rf_reg_pred = rf_reg.predict(X_reg_test)
evaluate_regression(y_reg_test, rf_reg_pred, "Random Forest Regression")

# SVR
svr_model = SVR()
svr_model.fit(X_reg_train, y_reg_train)
svr_pred = svr_model.predict(X_reg_test)
evaluate_regression(y_reg_test, svr_pred, "SVR")

################### Additional Evaluation Metrics ###################
def evaluate_regression_extended(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Extended Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

# For Classification Models
def evaluate_classification(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_class_train, y_class_train)
log_reg_pred = log_reg.predict(X_class_test)
evaluate_classification(y_class_test, log_reg_pred, "Logistic Regression")

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_class_train, y_class_train)
dt_clf_pred = dt_clf.predict(X_class_test)
evaluate_classification(y_class_test, dt_clf_pred, "Decision Tree")

# Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_class_train, y_class_train)
rf_clf_pred = rf_clf.predict(X_class_test)
evaluate_classification(y_class_test, rf_clf_pred, "Random Forest")

# SVM Classifier
svm_clf = SVC(random_state=42)
svm_clf.fit(X_class_train, y_class_train)
svm_clf_pred = svm_clf.predict(X_class_test)
evaluate_classification(y_class_test, svm_clf_pred, "SVM")

# KNN Classifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_class_train, y_class_train)
knn_clf_pred = knn_clf.predict(X_class_test)
evaluate_classification(y_class_test, knn_clf_pred, "KNN")

# Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(X_class_train, y_class_train)
nb_clf_pred = nb_clf.predict(X_class_test)
evaluate_classification(y_class_test, nb_clf_pred, "Naive Bayes")

# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_class_train, y_class_train)
gb_clf_pred = gb_clf.predict(X_class_test)
evaluate_classification(y_class_test, gb_clf_pred, "Gradient Boosting")

################### Additional Evaluation Metrics ###################
def evaluate_classification_extended(y_true, y_pred, y_prob, model_name):
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend()
    
    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.tight_layout()
    plt.show()

################### Visualize Model Comparisons ###################
# Plot regression model performance
plt.figure(figsize=(10, 6))
regression_models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR']
rmse_scores = [
    np.sqrt(mean_squared_error(y_reg_test, lr_pred)),
    np.sqrt(mean_squared_error(y_reg_test, dt_reg_pred)),
    np.sqrt(mean_squared_error(y_reg_test, rf_reg_pred)),
    np.sqrt(mean_squared_error(y_reg_test, svr_pred))
]
plt.bar(regression_models, rmse_scores)
plt.title('Regression Models Performance Comparison')
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot classification model performance
plt.figure(figsize=(10, 6))
classification_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                        'SVM', 'KNN', 'Naive Bayes', 'Gradient Boosting']
accuracy_scores = [
    accuracy_score(y_class_test, log_reg_pred),
    accuracy_score(y_class_test, dt_clf_pred),
    accuracy_score(y_class_test, rf_clf_pred),
    accuracy_score(y_class_test, svm_clf_pred),
    accuracy_score(y_class_test, knn_clf_pred),
    accuracy_score(y_class_test, nb_clf_pred),
    accuracy_score(y_class_test, gb_clf_pred)
]
plt.bar(classification_models, accuracy_scores)
plt.title('Classification Models Performance Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

################### Cross Validation ###################
def perform_cross_validation(model, X, y, model_name, is_regression=True):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    if is_regression:
        scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
        print(f"\n{model_name} Cross-Validation Results:")
        print(f"Average RMSE: {-scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    else:
        scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        print(f"\n{model_name} Cross-Validation Results:")
        print(f"Average Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

################### Genetic Algorithm for Feature Selection ###################
from deap import base, creator, tools, algorithms
import random

def genetic_algorithm_feature_selection(X, y, model, n_generations=50):
    # Create types for genetic algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_bool, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evalFeatures(individual):
        # Get selected features
        cols = [i for i, v in enumerate(individual) if v == 1]
        if len(cols) == 0:
            return 0.0,
        
        # Get subset of features
        X_selected = X.iloc[:, cols]
        
        # Perform cross validation
        cv_scores = cross_val_score(model, X_selected, y, cv=5, 
                                  scoring='accuracy' if isinstance(model, ClassifierMixin) else 'neg_root_mean_squared_error')
        return np.mean(cv_scores),
    
    toolbox.register("evaluate", evalFeatures)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create population and run algorithm
    population = toolbox.population(n=50)
    result, logbook = algorithms.eaSimple(population, toolbox, 
                                        cxpb=0.7, mutpb=0.2, 
                                        ngen=n_generations, 
                                        verbose=False)
    
    return tools.selBest(result, k=1)[0]

################### Model Interpretation ###################
def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print(f"Model {model_name} doesn't support feature importance")
        return
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importances)
    plt.title(f'{model_name} - Feature Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Apply cross-validation to all models
perform_cross_validation(rf_clf, X_class_train, y_class_train, "Random Forest", is_regression=False)
perform_cross_validation(lr_model, X_reg_train, y_reg_train, "Linear Regression", is_regression=True)

# Apply GA feature selection
best_features = genetic_algorithm_feature_selection(X_reg, y_reg, rf_reg)
print("Best features selected by GA:", best_features)

# Plot feature importance for interpretable models
plot_feature_importance(rf_clf, X_reg.columns, "Random Forest Classifier")