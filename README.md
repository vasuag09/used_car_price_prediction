# 🚗 Used Car Price Prediction

Predict the selling price of used cars using machine learning.

## 📌 Project Overview

This project uses regression models to predict the price of used cars based on various features such as brand, fuel type, transmission, kilometers driven, engine size, and more. The goal is to help users estimate the resale value of cars using real-world data.

## 📊 Dataset

- Source: [CarDekho (Kaggle)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
- Records: ~6,000 rows
- Features:
  - Name, Brand, Location, Year
  - Fuel Type, Transmission, Owner Type
  - Mileage, Engine, Power, Seats
  - Selling Price (target)

## 🧹 Data Preprocessing

- Removed irrelevant/missing-heavy columns (`New_Price`)
- Extracted numeric values from `Mileage`, `Engine`, and `Power`
- Handled missing values using median imputation
- Feature engineered:
  - `Car_Age = Current Year - Year`
  - `Brand` from `Name`
- One-hot encoded categorical variables with `drop='first'`
- Scaled numerical columns using `StandardScaler`

## ⚙️ Model Training

Tested multiple models:
- ✅ **Linear Regression**
- ✅ **Polynomial Regression**
- ✅ **Random Forest Regressor**
- ✅ **XGBoost Regressor** (**Best**)

### ✅ Final Model: XGBoost

```python
XGBRegressor(
  n_estimators=1000,
  learning_rate=0.05,
  max_depth=6,
  subsample=0.9,
  colsample_bytree=0.9,
  random_state=42
)
