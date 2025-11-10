import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load data
data = pd.read_csv('../../../Data/trainingData/trainingSample.csv')
data = data.replace(['-9999', -9999], np.nan)
output = 'output/'
os.makedirs(output, exist_ok=True)

print("=" * 80)
print("TRAINING DATA OVERVIEW")
print("=" * 80)
print(f"\nDataset shape: {data.shape}")
print(f"\nColumn names:\n{data.columns.tolist()}")
print(f"\nData types:\n{data.dtypes}")
print(f"\nMissing values:\n{data.isnull().sum()}")

# Identify target column
target_col = 'target'

# Define categorical columns explicitly
categorical_cols = ['Land_cover', 'ALC_old', 'Profile_depth', 'CaCO3_rank', 'Texture_group', 
                    'Aggregate_texture', 'Aquifers', 'bedrock']

# ============================================================================
# SEPARATE CATEGORICAL AND NUMERICAL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE TYPE IDENTIFICATION")
print("=" * 80)

# Get all feature columns (excluding target)
feature_cols = [col for col in data.columns if col != target_col]

# Identify numerical columns (all features except the defined categorical ones)
numerical_cols = [col for col in feature_cols if col not in categorical_cols]

print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"\nNumerical features ({len(numerical_cols)}): {numerical_cols}")

# Check if categorical columns exist in the dataset
missing_cat_cols = [col for col in categorical_cols if col not in data.columns]
if missing_cat_cols:
    print(f"\nWarning: These categorical columns are not in the dataset: {missing_cat_cols}")
    categorical_cols = [col for col in categorical_cols if col in data.columns]

# Display unique values for categorical columns
print("\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"  {col}: {data[col].nunique()} unique values - {data[col].unique()[:10]}")

# Check for missing values in each feature type
print("\n" + "=" * 80)
print("MISSING VALUES ANALYSIS")
print("=" * 80)

if len(numerical_cols) > 0:
    num_missing = data[numerical_cols].isnull().sum()
    num_missing = num_missing[num_missing > 0]
    if len(num_missing) > 0:
        print(f"\nNumerical columns with missing values:")
        print(num_missing)
        print(f"\nTotal missing in numerical features: {data[numerical_cols].isnull().sum().sum()}")
    else:
        print("\nNo missing values in numerical features")

if len(categorical_cols) > 0:
    cat_missing = data[categorical_cols].isnull().sum()
    cat_missing = cat_missing[cat_missing > 0]
    if len(cat_missing) > 0:
        print(f"\nCategorical columns with missing values:")
        print(cat_missing)
        print(f"\nTotal missing in categorical features: {data[categorical_cols].isnull().sum().sum()}")
    else:
        print("\nNo missing values in categorical features")

# Handle missing values
print("\n" + "=" * 80)
print("HANDLING MISSING VALUES")
print("=" * 80)

# Create a copy for analysis
data_clean = data.copy()

# For numerical columns: impute with median
if len(numerical_cols) > 0:
    for col in numerical_cols:
        if data_clean[col].isnull().sum() > 0:
            median_val = data_clean[col].median()
            data_clean[col].fillna(median_val, inplace=True)
            print(f"Imputed {col} with median: {median_val:.4f}")

# For categorical columns: impute with mode or 'Unknown'
if len(categorical_cols) > 0:
    for col in categorical_cols:
        if data_clean[col].isnull().sum() > 0:
            mode_val = data_clean[col].mode()
            if len(mode_val) > 0:
                data_clean[col].fillna(mode_val[0], inplace=True)
                print(f"Imputed {col} with mode: {mode_val[0]}")
            else:
                data_clean[col].fillna('Unknown', inplace=True)
                print(f"Imputed {col} with 'Unknown'")

print(f"\nMissing values after imputation: {data_clean.isnull().sum().sum()}")

# Use cleaned data for the rest of the analysis
data = data_clean

# ============================================================================
# CLASS DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)

class_counts = data[target_col].value_counts().sort_index()
class_props = data[target_col].value_counts(normalize=True).sort_index()

print(f"\nClass counts:\n{class_counts}")
print(f"\nClass proportions:\n{class_props}")

# Plot class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
class_counts.plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Class Distribution (Counts)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)

class_props.plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Class Distribution (Proportions)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Proportion')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('output/class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# NUMERICAL FEATURES ANALYSIS
# ============================================================================
if len(numerical_cols) > 0:
    print("\n" + "=" * 80)
    print("NUMERICAL FEATURES - CLASS SEPARABILITY")
    print("=" * 80)
    
    # Feature statistics by class
    print("\nNumerical feature statistics by class:")
    num_stats = data.groupby(target_col)[numerical_cols].agg(['mean', 'std'])
    print(num_stats)
    
    # 1. Fisher Score for numerical features
    def fisher_score(feature, target):
        """Calculate Fisher score for a feature"""
        classes = target.unique()
        overall_mean = feature.mean()
        
        numerator = 0
        denominator = 0
        
        for c in classes:
            class_data = feature[target == c]
            n_c = len(class_data)
            mean_c = class_data.mean()
            var_c = class_data.var()
            
            numerator += n_c * (mean_c - overall_mean) ** 2
            denominator += n_c * var_c
        
        return numerator / (denominator + 1e-10)
    
    fisher_scores = {}
    for col in numerical_cols:
        fisher_scores[col] = fisher_score(data[col], data[target_col])
    
    fisher_df = pd.DataFrame.from_dict(fisher_scores, orient='index', columns=['Fisher Score'])
    fisher_df = fisher_df.sort_values('Fisher Score', ascending=False)
    
    print("\nTop 10 numerical features by Fisher Score (higher = better separability):")
    print(fisher_df.head(10))
    
    # Plot top features
    plt.figure(figsize=(12, 6))
    fisher_df.head(min(15, len(fisher_df))).plot(kind='barh', legend=False, color='teal')
    plt.xlabel('Fisher Score', fontsize=12)
    plt.title('Numerical Features by Fisher Score', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('output/numerical_fisher_scores.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. ANOVA F-statistic for numerical features
    from sklearn.feature_selection import f_classif
    
    X_num = data[numerical_cols].values
    y = data[target_col].values
    
    f_stats, p_values = f_classif(X_num, y)
    anova_df = pd.DataFrame({
        'Feature': numerical_cols,
        'F-statistic': f_stats,
        'p-value': p_values
    }).sort_values('F-statistic', ascending=False)
    
    print("\nTop 10 numerical features by ANOVA F-statistic:")
    print(anova_df.head(10))
    
    # 3. Distribution comparison for top numerical features
    print("\n" + "=" * 80)
    print("NUMERICAL FEATURE DISTRIBUTIONS BY CLASS")
    print("=" * 80)
    
    top_num_features = fisher_df.head(min(6, len(fisher_df))).index.tolist()
    
    if len(top_num_features) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(top_num_features):
            for class_label in data[target_col].unique():
                class_data = data[data[target_col] == class_label][feature]
                axes[idx].hist(class_data, alpha=0.6, label=f'Class {class_label}', bins=30)
            
            axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
        
        # Hide empty subplots
        for idx in range(len(top_num_features), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('output/numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Box plots for top features
    if len(top_num_features) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(top_num_features):
            data.boxplot(column=feature, by=target_col, ax=axes[idx])
            axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Class')
            axes[idx].set_ylabel('Value')
            plt.sca(axes[idx])
            plt.xticks(rotation=0)
        
        # Hide empty subplots
        for idx in range(len(top_num_features), 6):
            axes[idx].axis('off')
        
        plt.suptitle('')  # Remove the automatic title
        plt.tight_layout()
        plt.savefig('output/numerical_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# CATEGORICAL FEATURES ANALYSIS
# ============================================================================
if len(categorical_cols) > 0:
    print("\n" + "=" * 80)
    print("CATEGORICAL FEATURES - CLASS SEPARABILITY")
    print("=" * 80)
    
    # Chi-square test for categorical features
    chi2_results = {}
    cramers_v_results = {}
    
    def cramers_v(confusion_matrix):
        """Calculate Cramér's V statistic for categorical association"""
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        min_dim = min(confusion_matrix.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))
    
    for col in categorical_cols:
        contingency_table = pd.crosstab(data[col], data[target_col])
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
        cramers = cramers_v(contingency_table.values)
        
        chi2_results[col] = {'chi2': chi2, 'p-value': p_val}
        cramers_v_results[col] = cramers
    
    chi2_df = pd.DataFrame(chi2_results).T.sort_values('chi2', ascending=False)
    cramers_df = pd.DataFrame.from_dict(cramers_v_results, orient='index', 
                                         columns=['Cramers V'])
    cramers_df = cramers_df.sort_values('Cramers V', ascending=False)
    
    print("\nCategorical features ranked by Chi-square statistic:")
    print(chi2_df)
    
    print("\nCategorical features ranked by Cramér's V (0=no association, 1=perfect):")
    print(cramers_df)
    
    # Plot Cramér's V
    plt.figure(figsize=(12, 6))
    cramers_df.plot(kind='barh', legend=False, color='mediumpurple')
    plt.xlabel("Cramér's V", fontsize=12)
    plt.title("Categorical Features by Cramér's V", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('output/categorical_cramers_v.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Distribution plots for top categorical features
    top_cat_features = cramers_df.head(min(6, len(cramers_df))).index.tolist()
    
    if len(top_cat_features) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(top_cat_features):
            contingency = pd.crosstab(data[feature], data[target_col], normalize='index')
            contingency.plot(kind='bar', stacked=False, ax=axes[idx], alpha=0.8)
            axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Category')
            axes[idx].set_ylabel('Proportion')
            axes[idx].legend(title='Class', bbox_to_anchor=(1.05, 1))
            axes[idx].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for idx in range(len(top_cat_features), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('output/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Heatmap for top categorical features
    if len(top_cat_features) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(top_cat_features):
            contingency = pd.crosstab(data[feature], data[target_col])
            sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu', 
                       ax=axes[idx], cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Class')
            axes[idx].set_ylabel('Category')
        
        # Hide empty subplots
        for idx in range(len(top_cat_features), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('output/categorical_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# PCA VISUALIZATION (Numerical features only)
# ============================================================================
if len(numerical_cols) > 0:
    print("\n" + "=" * 80)
    print("PCA VISUALIZATION (Numerical Features)")
    print("=" * 80)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[numerical_cols])
    y = data[target_col].values
    
    # Apply PCA
    pca = PCA(n_components=min(2, len(numerical_cols)))
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Plot PCA
    plt.figure(figsize=(10, 8))
    for class_label in np.unique(y):
        mask = y == class_label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                    label=f'Class {class_label}', alpha=0.6, s=30)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    if len(pca.explained_variance_ratio_) > 1:
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title('PCA Visualization of Class Separability (Numerical Features)', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # PCA loadings (feature importance)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=numerical_cols
    )
    
    print("\nTop contributing features to PC1:")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(10))

# ============================================================================
# LDA VISUALIZATION (Numerical features only, if more than 2 classes)
# ============================================================================
if len(numerical_cols) > 0 and len(np.unique(data[target_col])) > 2:
    print("\n" + "=" * 80)
    print("LDA VISUALIZATION (Numerical Features)")
    print("=" * 80)
    
    n_components = min(len(np.unique(data[target_col])) - 1, 2, len(numerical_cols))
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X_scaled, y)
    
    print(f"\nExplained variance ratio: {lda.explained_variance_ratio_}")
    
    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        for class_label in np.unique(y):
            mask = y == class_label
            plt.scatter(X_lda[mask, 0], X_lda[mask, 1], 
                        label=f'Class {class_label}', alpha=0.6, s=30)
        
        plt.xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.title('LDA Visualization of Class Separability (Numerical Features)', 
                  fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/lda_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# CORRELATION WITH TARGET (for binary classification and numerical features)
# ============================================================================
if len(numerical_cols) > 0 and len(np.unique(data[target_col])) == 2:
    print("\n" + "=" * 80)
    print("POINT-BISERIAL CORRELATION (Numerical Features)")
    print("=" * 80)
    
    correlations = {}
    for col in numerical_cols:
        corr, p_val = stats.pointbiserialr(data[target_col], data[col])
        correlations[col] = {'correlation': corr, 'p-value': p_val}
    
    corr_df = pd.DataFrame(correlations).T.sort_values('correlation', 
                                                        key=abs, ascending=False)
    print("\nTop 10 numerical features by correlation with target:")
    print(corr_df.head(10))

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("CLASS SEPARABILITY SUMMARY")
print("=" * 80)

summary = f"""
Dataset Summary:
- Total samples: {len(data)}
- Number of numerical features: {len(numerical_cols)}
- Number of categorical features: {len(categorical_cols)}
- Number of classes: {len(np.unique(data[target_col]))}
- Class balance: {'Balanced' if class_props.max() - class_props.min() < 0.1 else 'Imbalanced'}
"""

if len(numerical_cols) > 0:
    summary += f"""
Top 5 Most Discriminative Numerical Features:
{fisher_df.head(5).to_string()}

PCA Analysis (Numerical Features):
- First {pca.n_components_} component(s) explain {pca.explained_variance_ratio_.sum():.2%} of variance
"""

if len(categorical_cols) > 0:
    summary += f"""
Top 5 Most Discriminative Categorical Features:
{cramers_df.head(5).to_string()}
"""

summary += f"""
Overall Recommendation:
- Numerical features: {'Strong separability detected' if len(numerical_cols) > 0 and fisher_df['Fisher Score'].mean() > 1.0 else 'Moderate overlap in numerical features' if len(numerical_cols) > 0 else 'No numerical features'}
- Categorical features: {'Strong association with target' if len(categorical_cols) > 0 and cramers_df['Cramers V'].mean() > 0.3 else 'Moderate association with target' if len(categorical_cols) > 0 else 'No categorical features'}
- XGBoost is well-suited for this mixed-type dataset and can handle both feature types effectively
"""

print(summary)