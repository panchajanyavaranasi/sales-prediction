import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
train_path = os.path.join(BASE_DIR, "data", "train.csv")

# -----------------------------
# Load Train Data
# -----------------------------
df = pd.read_csv(train_path)

# -----------------------------
# Drop unnecessary columns
# -----------------------------
df = df.drop(["id", "date"], axis=1)

# -----------------------------
# Split Features & Target
# -----------------------------
X = df.drop("sales_revenue", axis=1)
y = df["sales_revenue"]

# -----------------------------
# Separate numerical & categorical
# -----------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

X_num = X[num_cols]
X_cat = X[cat_cols]

# -----------------------------
# Handle Missing Values
# -----------------------------
imputer = SimpleImputer(strategy="mean")
X_num_imputed = pd.DataFrame(imputer.fit_transform(X_num), columns=num_cols)

# -----------------------------
# Encode categorical
# -----------------------------
X_cat_encoded = pd.get_dummies(X_cat)

# -----------------------------
# Combine
# -----------------------------
X_final = pd.concat([X_num_imputed, X_cat_encoded], axis=1)

# Save columns
columns = X_final.columns

# -----------------------------
# Split (internal validation)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -----------------------------
# Model
# -----------------------------
model = KNeighborsRegressor(n_neighbors=20)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Save artifacts
# -----------------------------
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

pickle.dump(model, open(os.path.join(models_dir, "knn_model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(models_dir, "scaler.pkl"), "wb"))
pickle.dump(imputer, open(os.path.join(models_dir, "imputer.pkl"), "wb"))
pickle.dump(columns, open(os.path.join(models_dir, "columns.pkl"), "wb"))

print("✅ Model training complete and saved!")