import os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ✅ Suppress XGBoost warnings
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
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return np.mean(preds == y_test)

# ✅ Run tuning
print("🔧 Starting Optuna Tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

# ✅ Train final model
best_params = study.best_params
best_params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ✅ Evaluation
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["No FoG", "FoG"], yticklabels=["No FoG", "FoG"])
plt.title("🧠 Freezing of Gait Detection - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


