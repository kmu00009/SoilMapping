import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif

# -------------------------------
# SETTINGS
# -------------------------------
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Paths
data_path = '../../../Data/trainingData/trainingSample.csv'
output = 'output_numerical/'
os.makedirs(output, exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
data = pd.read_csv(data_path)
data = data.replace(['-9999', -9999], np.nan)
target_col = 'target'

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Missing values:\n{data.isnull().sum()}")

# -------------------------------
# IDENTIFY NUMERICAL COLUMNS
# -------------------------------
numerical_cols = [col for col in data.columns if col != target_col]
print(f"\nAll numerical features ({len(numerical_cols)}): {numerical_cols}")

# -------------------------------
# HANDLE MISSING VALUES
# -------------------------------
data_clean = data.copy()
for col in numerical_cols:
    if data_clean[col].isnull().sum() > 0:
        median_val = data_clean[col].median()
        data_clean[col].fillna(median_val, inplace=True)
        print(f"Imputed {col} with median: {median_val:.4f}")

data = data_clean
print(f"\nMissing values after imputation: {data.isnull().sum().sum()}")

# -------------------------------
# CLASS DISTRIBUTION
# -------------------------------
class_counts = data[target_col].value_counts().sort_index()
class_props = data[target_col].value_counts(normalize=True).sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
class_counts.plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Class Distribution (Counts)')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)

class_props.plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Class Distribution (Proportions)')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Proportion')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(output, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# FISHER SCORE
# -------------------------------
def fisher_score(feature, target):
    classes = target.unique()
    overall_mean = feature.mean()
    numerator, denominator = 0, 0
    for c in classes:
        class_data = feature[target == c]
        n_c = len(class_data)
        mean_c = class_data.mean()
        var_c = class_data.var()
        numerator += n_c * (mean_c - overall_mean)**2
        denominator += n_c * var_c
    return numerator / (denominator + 1e-10)

fisher_scores = {col: fisher_score(data[col], data[target_col]) for col in numerical_cols}
fisher_df = pd.DataFrame.from_dict(fisher_scores, orient='index', columns=['Fisher Score']).sort_values('Fisher Score', ascending=False)
print("\nTop 10 numerical features by Fisher Score:")
print(fisher_df.head(10))

# Plot top Fisher features
plt.figure(figsize=(12, 6))
fisher_df.head(min(15, len(fisher_df))).plot(kind='barh', legend=False, color='teal')
plt.xlabel('Fisher Score')
plt.title('Numerical Features by Fisher Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output, 'numerical_fisher_scores.png'), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# ANOVA F-STATISTICS
# -------------------------------
X_num = data[numerical_cols].values
y = data[target_col].values
f_stats, p_values = f_classif(X_num, y)
anova_df = pd.DataFrame({'Feature': numerical_cols, 'F-statistic': f_stats, 'p-value': p_values}).sort_values('F-statistic', ascending=False)
print("\nTop 10 numerical features by ANOVA F-statistic:")
print(anova_df.head(10))

# -------------------------------
# DISTRIBUTION PLOTS FOR TOP FEATURES
# -------------------------------
top_features = fisher_df.head(min(6, len(fisher_df))).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    for class_label in data[target_col].unique():
        class_data = data[data[target_col] == class_label][feature]
        axes[idx].hist(class_data, alpha=0.6, label=f'Class {class_label}', bins=30)
    axes[idx].set_title(f'{feature}')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

for idx in range(len(top_features), 6):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output, 'numerical_distributions.png'), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# BOX PLOTS FOR TOP FEATURES
# -------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()
for idx, feature in enumerate(top_features):
    data.boxplot(column=feature, by=target_col, ax=axes[idx])
    axes[idx].set_title(f'{feature}')
    axes[idx].set_xlabel('Class')
    axes[idx].set_ylabel('Value')
    plt.sca(axes[idx])
    plt.xticks(rotation=0)

for idx in range(len(top_features), 6):
    axes[idx].axis('off')

plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(output, 'numerical_boxplots.png'), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# PCA VISUALIZATION
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[numerical_cols])
pca = PCA(n_components=min(2, len(numerical_cols)))
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
for class_label in np.unique(y):
    mask = y == class_label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {class_label}', alpha=0.6, s=30)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
if len(pca.explained_variance_ratio_) > 1:
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA Visualization of Class Separability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output, 'pca_visualization.png'), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# LDA VISUALIZATION (if more than 2 classes)
# -------------------------------
if len(np.unique(y)) > 2:
    n_components = min(len(np.unique(y)) - 1, 2, len(numerical_cols))
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X_scaled, y)

    plt.figure(figsize=(10, 8))
    for class_label in np.unique(y):
        mask = y == class_label
        plt.scatter(X_lda[mask, 0], X_lda[mask, 1], label=f'Class {class_label}', alpha=0.6, s=30)

    plt.xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.2%} variance)')
    if n_components >= 2:
        plt.ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('LDA Visualization of Class Separability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'lda_visualization.png'), dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------
# POINT-BISERIAL CORRELATION (binary target only)
# -------------------------------
if len(np.unique(y)) == 2:
    correlations = {}
    for col in numerical_cols:
        corr, p_val = stats.pointbiserialr(data[target_col], data[col])
        correlations[col] = {'correlation': corr, 'p-value': p_val}
    corr_df = pd.DataFrame(correlations).T.sort_values('correlation', key=abs, ascending=False)
    print("\nTop 10 numerical features by correlation with target:")
    print(corr_df.head(10))
