# 📊 Sales Revenue Prediction using KNN Regression

## 🚀 Project Overview

This project builds a **machine learning system** to predict **sales revenue** based on marketing campaign data.

It uses a **K-Nearest Neighbors (KNN) Regression model** and is deployed using **Streamlit**, allowing users to input campaign parameters and get real-time predictions.

---

## 🎯 Objectives

* Predict sales revenue using marketing features
* Understand the impact of different campaign factors
* Handle real-world data challenges (missing values, categorical data)
* Build and deploy an interactive ML application

---

## 🧠 Key Concepts

### 🔹 KNN Regression

KNN predicts values by identifying the **K nearest data points** using distance metrics and averaging their outputs.

### 🔹 Feature Scaling

Standardization ensures all features contribute equally to distance calculation.

### 🔹 Missing Value Handling

Used **SimpleImputer** to fill missing numerical values.

### 🔹 Categorical Encoding

Applied **One-Hot Encoding** to convert categorical variables into numerical form.

---

## 🏗️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit

---

## 📂 Project Structure

```
Sales_prediction/
│
├── data/                  # Dataset (optional / ignored in production)
│   ├── train.csv
│   ├── test.csv
│
├── src/
│   └── train_model.py     # Model training script
│
├── models/
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   ├── imputer.pkl
│   ├── columns.pkl
│
├── app/
│   └── app.py             # Streamlit application
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. Data preprocessing (cleaning, encoding, imputation)
2. Train KNN regression model
3. Save model artifacts (`.pkl` files)
4. Streamlit app loads model
5. User inputs → prediction generated

---

## ▶️ Run Locally

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python src/train_model.py
```

### 3. Run the application

```
streamlit run app/app.py
```

---

## 🌐 Deployment

The application can be deployed using **Streamlit Cloud**.

👉 https://sales-prediction-r7fe3dq6tvcol9lhx3si9n.streamlit.app/

---

## 📈 Model Pipeline

```
User Input
   ↓
Imputer (Handle missing values)
   ↓
Encoding (Categorical → Numeric)
   ↓
Scaling (StandardScaler)
   ↓
KNN Model
   ↓
Predicted Revenue
```

---

## 🧪 Testing & Validation

* Tested with various input combinations
* Verified model stability for edge cases
* Ensured consistent preprocessing during prediction

---

## 💡 Key Learnings

* End-to-end ML pipeline development
* Importance of preprocessing and scaling
* Handling categorical and missing data
* Deploying ML models using Streamlit
* Version control using Git & GitHub

---

## 🚀 Future Enhancements

* Hyperparameter tuning (optimal K value)
* Model comparison (Linear Regression, Random Forest)
* Visualization dashboard (charts & insights)
* Cloud deployment (AWS / Docker)

---

## 👤 Author

**Panchajanya Varanasi**

---

## ⭐ Acknowledgment

This project was developed as part of a machine learning learning initiative, focusing on practical implementation and deployment.
