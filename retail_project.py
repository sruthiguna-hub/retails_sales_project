import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")
# -------------------------------
# Dataset Information
# -------------------------------
print("First 5 Rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nDataset Shape:", df.shape)
# -------------------------------
# Data Cleaning
# -------------------------------
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)
# -------------------------------
# Sales Analysis
# -------------------------------
print("\n===== SALES ANALYSIS =====")
print("Total Sales:", df["Sales"].sum())
print("Total Profit:", df["Profit"].sum())
print("Average Sales:", df["Sales"].mean())
print("Highest Sales:", df["Sales"].max())
print("Lowest Sales:", df["Sales"].min())
print("Average Profit:", df["Profit"].mean())
# -------------------------------
# Graph 1 - Sales by Category
# -------------------------------
category_sales = df.groupby("Category")["Sales"].sum()
plt.figure(figsize=(8,5))
category_sales.plot(kind="bar")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("output/category_sales.png")
plt.close()
print("Category Sales graph saved successfully.")
# -------------------------------
# Graph 2 - Sales by Region
# -------------------------------
region_sales = df.groupby("Region")["Sales"].sum()
plt.figure(figsize=(8,5))
region_sales.plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("output/region_sales.png")
plt.close()
print("Region Sales graph saved successfully.")
# -------------------------------
# Graph 3 - Monthly Sales Trend
# -------------------------------
# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])
# Extract Month Name
df["Month"] = df["Order Date"].dt.month_name()
# Correct Month Order
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
monthly_sales = (
    df.groupby("Month")["Sales"]
      .sum()
      .reindex(month_order)
)
plt.figure(figsize=(10,5))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/monthly_sales.png")
plt.close()
print("Monthly Sales graph saved successfully.")
# -------------------------------
# Graph 4 - Top 10 Customers by Sales
# -------------------------------
top_customers = (
    df.groupby("Customer Name")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)
plt.figure(figsize=(10,5))
top_customers.plot(kind="bar")
plt.title("Top 10 Customers by Sales")
plt.xlabel("Customer Name")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("output/top_customers.png")
plt.close()
print("Top 10 Customers graph saved successfully.")
# -------------------------------
# Machine Learning - Sales Prediction
# -------------------------------
# Features (Input)
X = df[["Quantity", "Discount", "Profit"]]
# Target (Output)
y = df["Sales"]
print("\nFeatures and Target selected successfully.")
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Dataset split successfully.")
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
# Train Model
model = LinearRegression()
model.fit(X_train, y_train)
print("Linear Regression model trained successfully.")
# Prediction
y_pred = model.predict(X_test)
print("Prediction completed successfully.")
# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n===== MODEL PERFORMANCE =====")
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)
# -------------------------------
# Graph 5 - Actual vs Predicted Sales
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.tight_layout()
plt.savefig("output/actual_vs_predicted.png")
plt.close()
print("Actual vs Predicted graph saved successfully.")
# -------------------------------
# Save Predictions to CSV
# -------------------------------
prediction_df = pd.DataFrame({
    "Actual Sales": y_test,
    "Predicted Sales": y_pred
})
prediction_df.to_csv("output/sales_predictions.csv", index=False)
print("Predictions saved successfully.")