import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data & Preprocessing
# ==========================================
print("Loading data...")
df = pd.read_csv('Air_Quality.csv')

# Drop columns that are completely missing (e.g., 'Message') or irrelevant identifiers
df = df.drop(columns=['Message', 'Unique ID', 'Geo Join ID', 'Indicator ID'])

# For simplicity, filter to include only actual atmospheric pollutants
pollutants = ['Nitrogen dioxide (NO2)', 'Fine particles (PM 2.5)', 'Ozone (O3)']
df = df[df['Name'].isin(pollutants)].copy()

# Missing Value Treatment
# Fill missing 'Data Value' with the median (if any exist)
df['Data Value'] = df['Data Value'].fillna(df['Data Value'].median())

# Feature Engineering: Extract Year and Month from Start_Date
df['Start_Date'] = pd.to_datetime(df['Start_Date'])
df['Year'] = df['Start_Date'].dt.year
df['Month'] = df['Start_Date'].dt.month
df = df.drop(columns=['Start_Date', 'Time Period'])

# Encode categorical variables (Environmental Parameters)
le_dict = {}
cat_cols = ['Name', 'Measure', 'Measure Info', 'Geo Type Name', 'Geo Place Name']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Normalization
scaler = StandardScaler()
features_to_scale = cat_cols + ['Year', 'Month', 'Data Value']
df_scaled = pd.DataFrame(scaler.fit_transform(df[features_to_scale]), columns=features_to_scale)

# ==========================================
# 2. Unsupervised Learning: Anomaly Detection
# ==========================================
print("\nPerforming Anomaly Detection using Gaussian Mixture Models...")
# Using GMM to find abnormal pollution patterns
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(df_scaled)
scores = gmm.score_samples(df_scaled)

# Define anomalies as the 5% of data with the lowest log-likelihood scores
threshold = np.percentile(scores, 5)
df['Anomaly'] = (scores < threshold).astype(int)
print(f"Detected {df['Anomaly'].sum()} abnormal pollution spikes (anomalies).")

# ==========================================
# 3. Create Pollution Categories
# ==========================================
# Create Low (0), Moderate (1), High (2) categories based on quantiles per pollutant
def categorize_pollution(series):
    q33 = series.quantile(0.33)
    q67 = series.quantile(0.67)
    # pd.cut handles the binning based on the calculated quantiles
    return pd.cut(series, bins=[-np.inf, q33, q67, np.inf], labels=[0, 1, 2])

df['Category'] = df.groupby('Name')['Data Value'].transform(categorize_pollution)

# ==========================================
# 4. Supervised Classifiers & Evaluation
# ==========================================
# Features (Environmental parameters) and Target (Pollution Category)
X = df.drop(columns=['Data Value', 'Category', 'Anomaly'])
y = df['Category'].astype(int)

# Split the FULL dataset (Contains anomalies)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}

def evaluate_models(models, X_train, y_train, X_test, y_test, scenario_name):
    print(f"\n--- Model Evaluation: {scenario_name} ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data & Preprocessing
# ==========================================
print("Loading data...")
df = pd.read_csv('Air_Quality.csv')

# Drop columns that are completely missing (e.g., 'Message') or irrelevant identifiers
df = df.drop(columns=['Message', 'Unique ID', 'Geo Join ID', 'Indicator ID'])

# For simplicity, filter to include only actual atmospheric pollutants
pollutants = ['Nitrogen dioxide (NO2)', 'Fine particles (PM 2.5)', 'Ozone (O3)']
df = df[df['Name'].isin(pollutants)].copy()

# Missing Value Treatment
# Fill missing 'Data Value' with the median (if any exist)
df['Data Value'] = df['Data Value'].fillna(df['Data Value'].median())

# Feature Engineering: Extract Year and Month from Start_Date
df['Start_Date'] = pd.to_datetime(df['Start_Date'])
df['Year'] = df['Start_Date'].dt.year
df['Month'] = df['Start_Date'].dt.month
df = df.drop(columns=['Start_Date', 'Time Period'])

# Encode categorical variables (Environmental Parameters)
le_dict = {}
cat_cols = ['Name', 'Measure', 'Measure Info', 'Geo Type Name', 'Geo Place Name']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Normalization
scaler = StandardScaler()
features_to_scale = cat_cols + ['Year', 'Month', 'Data Value']
df_scaled = pd.DataFrame(scaler.fit_transform(df[features_to_scale]), columns=features_to_scale)

# ==========================================
# 2. Unsupervised Learning: Anomaly Detection
# ==========================================
print("\nPerforming Anomaly Detection using Gaussian Mixture Models...")
# Using GMM to find abnormal pollution patterns
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(df_scaled)
scores = gmm.score_samples(df_scaled)

# Define anomalies as the 5% of data with the lowest log-likelihood scores
threshold = np.percentile(scores, 5)
df['Anomaly'] = (scores < threshold).astype(int)
print(f"Detected {df['Anomaly'].sum()} abnormal pollution spikes (anomalies).")

# ==========================================
# 3. Create Pollution Categories
# ==========================================
# Create Low (0), Moderate (1), High (2) categories based on quantiles per pollutant
def categorize_pollution(series):
    q33 = series.quantile(0.33)
    q67 = series.quantile(0.67)
    # pd.cut handles the binning based on the calculated quantiles
    return pd.cut(series, bins=[-np.inf, q33, q67, np.inf], labels=[0, 1, 2])

df['Category'] = df.groupby('Name')['Data Value'].transform(categorize_pollution)

# ==========================================
# 4. Supervised Classifiers & Evaluation
# ==========================================
# Features (Environmental parameters) and Target (Pollution Category)
X = df.drop(columns=['Data Value', 'Category', 'Anomaly'])
y = df['Category'].astype(int)

# Split the FULL dataset (Contains anomalies)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}

def evaluate_models(models, X_train, y_train, X_test, y_test, scenario_name):
    print(f"\n--- Model Evaluation: {scenario_name} ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

# Evaluate on dataset WITH anomalies
evaluate_models(models, X_train, y_train, X_test, y_test, "Trained on FULL Data (Includes Anomalies)")

# ==========================================
# 5. Investigate Anomaly Detection Robustness
# ==========================================
# To see how anomaly detection improves robustness, we remove anomalies from the training set.
# By training only on "normal" data, the models learn the underlying true patterns better without noise.

df_normal = df[df['Anomaly'] == 0]
X_norm = df_normal.drop(columns=['Data Value', 'Category', 'Anomaly'])
y_norm = df_normal['Category'].astype(int)

# Split the CLEAN dataset
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

# Evaluate models trained on CLEAN data
evaluate_models(models, X_train_n, y_train_n, X_test_n, y_test_n, "Trained on CLEAN Data (Anomalies Removed)")

print("\nConclusion: Comparing the two scenarios above investigates robustness. Often, removing severe anomalies allows classifiers (especially sensitive ones like Neural Networks) to define clearer decision boundaries, improving F1-scores and predictive accuracy on normal environmental parameters.")        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

# Evaluate on dataset WITH anomalies
evaluate_models(models, X_train, y_train, X_test, y_test, "Trained on FULL Data (Includes Anomalies)")

# ==========================================
# 5. Investigate Anomaly Detection Robustness
# ==========================================
# To see how anomaly detection improves robustness, we remove anomalies from the training set.
# By training only on "normal" data, the models learn the underlying true patterns better without noise.

df_normal = df[df['Anomaly'] == 0]
X_norm = df_normal.drop(columns=['Data Value', 'Category', 'Anomaly'])
y_norm = df_normal['Category'].astype(int)

# Split the CLEAN dataset
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

# Evaluate models trained on CLEAN data
evaluate_models(models, X_train_n, y_train_n, X_test_n, y_test_n, "Trained on CLEAN Data (Anomalies Removed)")

print("\nConclusion: Comparing the two scenarios above investigates robustness. Often, removing severe anomalies allows classifiers (especially sensitive ones like Neural Networks) to define clearer decision boundaries, improving F1-scores and predictive accuracy on normal environmental parameters.")
