# fraud_analysis.py

# How does the transaction Amount vary between fraudulent and non-fraudulent transactions?
# What is the distribution of Time across fraud and non-fraud classes?
# How do V1 to V28 behave differently for Class 0 and Class 1? (e.g., boxplots of individual features against class)
# Is there any significant correlation between Amount and any of the PCA components for fraud cases?
# What is the correlation between each PCA feature and the target class?

# Can PCA components (V1–V28) together distinguish fraud from non-fraud (e.g., via t-SNE or UMAP)?
# Can a clustering algorithm (e.g., KMeans, DBSCAN) separate fraud cases from normal ones in the PCA-reduced space?
# Which features are most important in classifying fraud using a Random Forest model?
# Can we train a logistic regression or decision tree classifier to distinguish the two classes? What are their AUC-PR scores?
# Does combining Time, Amount, and key PCA features improve classification performance compared to using PCA alone?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load the dataset
file_path = 'creditcard.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find the dataset at {file_path}")

try:
    df = pd.read_csv(file_path, delimiter=',')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp1252', delimiter=',')


# Create a directory for saving plots
os.makedirs("static/plots", exist_ok=True)

# 1. Amount variation
plt.figure(figsize=(6, 4))
sns.boxplot(x='Class', y='Amount', data=df, palette=['green', 'red'])
plt.title('Transaction Amount by Class')
plt.savefig('static/plots/bivariate_amount.png')
plt.close()

# 2. Time distribution
plt.figure(figsize=(10, 4))
sns.histplot(data=df, x='Time', hue='Class', bins=100, element='step', stat='density', palette=['green', 'red'])
plt.title('Transaction Time Distribution by Class')
plt.savefig('static/plots/bivariate_time.png')
plt.close()

# 3. Boxplots V1–V28 vs Class
for col in [f'V{i}' for i in range(1, 29)]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Class', y=col, data=df, palette=['green', 'red'])
    plt.title(f'{col} by Class')
    plt.savefig(f'static/plots/bivariate_{col}.png')
    plt.close()

# 4. Correlation Amount and PCA in frauds
fraud_df = df[df['Class'] == 1]
fraud_corr = fraud_df.corr()['Amount'][[f'V{i}' for i in range(1, 29)]]
plt.figure(figsize=(10, 5))
fraud_corr.plot(kind='bar', color='red')
plt.title('Correlation Between Amount and PCA Features in Fraudulent Transactions')
plt.ylabel('Correlation Coefficient')
plt.savefig('static/plots/bivariate_amount_corr_fraud.png')
plt.close()

# 5. Correlation between PCA and Class
correlation = df.corr()['Class'][[f'V{i}' for i in range(1, 29)]]
plt.figure(figsize=(10, 5))
correlation.plot(kind='bar', color='blue')
plt.title('Correlation Between PCA Features and Class')
plt.ylabel('Correlation Coefficient')
plt.savefig('static/plots/bivariate_class_corr.png')
plt.close()

# Multivariate Analysis
X = df.drop(columns=['Class'])
y = df['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_pca = df[[f'V{i}' for i in range(1, 29)]]

# 1. t-SNE (updated: replaced 'n_iter' with 'max_iter')
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(X_pca[:10000])  # Subsample for speed
plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=y[:10000], palette=['green', 'red'], legend='full')
plt.title("t-SNE of PCA Components")
plt.savefig('static/plots/multivariate_tsne.png')
plt.close()

# 2. Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca[:10000])
plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=kmeans_labels, palette='Set1')
plt.title("KMeans Clustering on t-SNE PCA")
plt.savefig('static/plots/multivariate_kmeans.png')
plt.close()

# 3. Random Forest feature importance
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
importances.nlargest(15).plot(kind='barh', color='orange')
plt.title("Feature Importance (Random Forest)")
plt.savefig('static/plots/multivariate_rf_importance.png')
print("Saved multivariate_rf_importance.png")
plt.close()

# 4. Logistic & Decision Tree with AUC-PR
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_score = average_precision_score(y_test, lr.predict_proba(X_test)[:, 1])

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_score = average_precision_score(y_test, dt.predict_proba(X_test)[:, 1])

with open("static/plots/multivariate_model_scores.txt", "w") as f:
    f.write(f"Logistic Regression AUPRC: {lr_score:.4f}\n")
    f.write(f"Decision Tree AUPRC: {dt_score:.4f}\n")
print("Saved multivariate_model_scores.txt")


# 5. Performance with Time, Amount, top PCA
top_features = ['Time', 'Amount'] + importances.nlargest(5).index.tolist()
X_top = df[top_features]
X_train_top, X_test_top, _, _ = train_test_split(X_top, y, stratify=y, random_state=42)
rf_top = RandomForestClassifier()
rf_top.fit(X_train_top, y_train)
top_score = average_precision_score(y_test, rf_top.predict_proba(X_test_top)[:, 1])

with open("static/plots/multivariate_model_scores.txt", "a") as f:
    f.write(f"Top Feature Model AUPRC: {top_score:.4f}\n")
print("Appended top features AUPRC to txt")
