import os
import numpy as np
import pandas as pd
import optuna
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ✅ Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ✅ Dataset path
folder_path = r"C:\Users\suman\Downloads\daphnet+freezing+of+gait\dataset_fog_release\dataset"

# ✅ Load all sensor files
def load_all_data(folder_path):
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            data = np.loadtxt(filepath)
            data_list.append(data)
    return np.vstack(data_list)

# ✅ Feature and label extraction using sliding window
def extract_features_labels(data, window_size=100, step=50):
    X, y = [], []
    for start in range(0, len(data) - window_size, step):
        end = start + window_size
        window = data[start:end, :-1]  # sensor columns
        label_window = data[start:end, -1]  # labels (FoG or not)
        X.append(window.flatten())
        y.append(1 if np.mean(label_window) > 0.5 else 0)
    return np.array(X), np.array(y)

# ✅ Preprocess dataset
raw_data = load_all_data(folder_path)
X, y = extract_features_labels(raw_data)
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Optuna objective
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 300),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "verbose": 0,
        "random_state": 42
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1)

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return np.mean(preds == y_test)

# ✅ Run tuning
print("🔧 Starting Optuna Tuning with CatBoost...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

# ✅ Train final model with best parameters
best_params = study.best_params
best_params.update({"verbose": 0, "random_state": 42})
final_model = CatBoostClassifier(**best_params)
final_model.fit(X_train, y_train)

# ✅ Predict & Evaluate
y_pred = final_model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["No FoG", "FoG"], yticklabels=["No FoG", "FoG"])
plt.title("🧠 CatBoost - Freezing of Gait Detection - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
