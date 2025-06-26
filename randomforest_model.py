import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns

# ================== SETTINGS ==================
DATA_DIR = r"C:\Users\suman\Downloads\daphnet+freezing+of+gait\dataset_fog_release\dataset"
MAX_ROWS_PER_FILE = 100000  # Adjust to 10000 for faster testing

# ============= STEP 1: LOAD DATA ==============
print("[INFO] Loading data files...")
data_files = glob(os.path.join(DATA_DIR, "*.txt"))
print(f"[INFO] Found {len(data_files)} data files.")

data_list = []
start_time = time.time()

for i, file in enumerate(data_files):
    try:
        print(f"[INFO] Loading file {i+1}/{len(data_files)}: {os.path.basename(file)}")
        df = pd.read_csv(file, sep=r'\s+', header=None, nrows=MAX_ROWS_PER_FILE)
        data_list.append(df)
    except Exception as e:
        print(f"[ERROR] Failed to read {file}: {e}")

print(f"[INFO] Loaded {len(data_list)} files successfully in {time.time() - start_time:.2f} seconds.")

# =========== STEP 2: CONCAT AND PROCESS ============
print("[INFO] Concatenating data...")
df_all = pd.concat(data_list, ignore_index=True)
print(f"[INFO] Total shape of dataset: {df_all.shape}")

# If labels are not present in data, simulate them
if df_all.shape[1] < 12:
    print("[INFO] Simulating FoG labels (binary)...")
    df_all['label'] = (np.random.rand(len(df_all)) < 0.1).astype(int)
else:
    df_all.rename(columns={df_all.shape[1]-1: 'label'}, inplace=True)

X = df_all.iloc[:, :-1]
y = df_all.iloc[:, -1]

# =========== STEP 3: SPLIT DATA ============
print("[INFO] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# =========== STEP 4: TRAIN RANDOM FOREST ============
print("[INFO] Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# =========== STEP 5: EVALUATE ============
def evaluate_model(model, name):
    print(f"\n[RESULT] Evaluation for {name}")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# Evaluate Random Forest
evaluate_model(rf_model, "Random Forest")
