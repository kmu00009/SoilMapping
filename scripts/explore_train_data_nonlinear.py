# nonlinear_separability_analysis.py
"""
Nonlinear Class Separability Analysis
-------------------------------------
This script complements 'explore_train_data.py' by focusing on nonlinear
relationships between features and class labels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Optional: UMAP (install with `pip install umap-learn`)
try:
    import umap
    umap_available = True
except ImportError:
    umap_available = False
    print("UMAP not installed. Skipping UMAP visualization.")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = '../../../Data/trainingData/trainingSample.csv'
TARGET_COL = 'target'
OUTPUT_DIR = 'output_nonlinear/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOAD AND CLEAN DATA
# =============================================================================
print("=" * 80)
print("LOADING AND CLEANING DATA")
print("=" * 80)

data = pd.read_csv(DATA_PATH)
data = data.replace(['-9999', -9999], np.nan)
print(f"Original shape: {data.shape}")

# Drop rows with missing target
data = data.dropna(subset=[TARGET_COL])
print(f"After dropping rows with missing target: {data.shape}")

# Identify categorical and numerical features
categorical_cols = ['Land_cover', 'Profile_depth', 'CaCO3_rank', 'Texture_group',
                    'Aggregate_texture', 'Aquifers', 'bedrock']
categorical_cols = [c for c in categorical_cols if c in data.columns]
numerical_cols = [c for c in data.columns if c not in categorical_cols + [TARGET_COL]]

# Drop rows with all NaN in features
data = data.dropna(subset=numerical_cols + categorical_cols, how='all')
print(f"After dropping fully empty rows: {data.shape}")

# Encode categorical features (for MI and visualizations)
encoded_data = data.copy()
for c in categorical_cols:
    encoded_data[c] = LabelEncoder().fit_transform(encoded_data[c].astype(str))

# Impute remaining missing numeric values with median
for c in numerical_cols:
    encoded_data[c] = encoded_data[c].fillna(encoded_data[c].median())

# =============================================================================
# MUTUAL INFORMATION ANALYSIS (Nonlinear Dependency)
# =============================================================================
print("\n" + "=" * 80)
print("MUTUAL INFORMATION ANALYSIS")
print("=" * 80)

X = encoded_data[numerical_cols + categorical_cols]
y = encoded_data[TARGET_COL].astype(int)

mi = mutual_info_classif(X, y, discrete_features=[c in categorical_cols for c in X.columns], random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi}).sort_values('Mutual Information', ascending=False)

print("\nTop 10 features by Mutual Information (nonlinear relevance):")
print(mi_df.head(10))

# Plot MI
plt.figure(figsize=(10, 6))
sns.barplot(y='Feature', x='Mutual Information', data=mi_df.head(15), hue='Feature', palette='viridis', legend=False)
plt.title('Top Features by Mutual Information (Nonlinear Dependence)', fontsize=14, fontweight='bold')
plt.xlabel('Mutual Information')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mutual_information.png'), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# t-SNE VISUALIZATION (Nonlinear Structure)
# =============================================================================
print("\n" + "=" * 80)
print("t-SNE VISUALIZATION (Nonlinear Class Structure)")
print("=" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(encoded_data[numerical_cols])

tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='tab10', s=30, alpha=0.7)
plt.title('t-SNE Projection: Nonlinear Class Separability', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Class')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_classes.png'), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# UMAP VISUALIZATION (if available)
# =============================================================================
if umap_available:
    print("\n" + "=" * 80)
    print("UMAP VISUALIZATION (Nonlinear Class Structure)")
    print("=" * 80)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette='tab10', s=30, alpha=0.7)
    plt.title('UMAP Projection: Nonlinear Class Separability', fontsize=14, fontweight='bold')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'umap_classes.png'), dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# SHALLOW DECISION TREE TEST (Nonlinear Separability Measure)
# =============================================================================
print("\n" + "=" * 80)
print("SHALLOW DECISION TREE SEPARABILITY TEST")
print("=" * 80)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
scores = cross_val_score(tree, X, y, cv=5, scoring='accuracy')

print(f"\nMean CV Accuracy (depth=3): {scores.mean():.3f} ± {scores.std():.3f}")
if scores.mean() > 0.8:
    print("→ Strong nonlinear separability detected.")
elif scores.mean() > 0.6:
    print("→ Moderate nonlinear separability.")
else:
    print("→ Weak or highly overlapping class boundaries.")

# =============================================================================
# SUMMARY
# =============================================================================
summary = f"""
Nonlinear Separability Summary
------------------------------
Samples: {len(data)}
Numerical features: {len(numerical_cols)}
Categorical features: {len(categorical_cols)}

Top 5 features by Mutual Information:
{mi_df.head(5).to_string(index=False)}

Decision Tree (depth=3) cross-val accuracy: {scores.mean():.3f}

Interpretation:
- Mutual Information measures nonlinear feature relevance.
- t-SNE and UMAP visualize complex class boundaries.
- Decision Tree accuracy provides a practical measure of nonlinear separability.
"""
print(summary)

with open(os.path.join(OUTPUT_DIR, 'nonlinear_summary.txt'), 'w') as f:
    f.write(summary)

print("\nAll results saved in:", os.path.abspath(OUTPUT_DIR))
