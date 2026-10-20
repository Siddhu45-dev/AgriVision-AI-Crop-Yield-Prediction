
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Data/crop_yield.csv")

# Create output folder if needed
os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid")

print("DATASET EXPLORATION")
print("-" * 40)

print("Dataset shape:", df.shape)
print("\nTop 10 crops by number of records:")
print(df["Crop"].value_counts().head(10))

print("\nRecords by season:")
print(df["Season"].value_counts())

print("\nRecords by state (top 10):")
print(df["State"].value_counts().head(10))

print("\nYield statistics:")
print(df["Yield"].describe())

print("\nChecking invalid values:")
for col in ["Area", "Production", "Annual_Rainfall",
            "Fertilizer", "Pesticide", "Yield"]:
    print(f"{col}: negative values = {(df[col] < 0).sum()}")

# 1. Yield distribution
plt.figure(figsize=(9, 5))
sns.histplot(df["Yield"], bins=50, kde=True)
plt.title("Crop Yield Distribution")
plt.xlabel("Yield")
plt.tight_layout()
plt.savefig("outputs/yield_distribution.png", dpi=150)
plt.close()

# 2. Top crops
plt.figure(figsize=(10, 6))
df["Crop"].value_counts().head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Crops in Dataset")
plt.xlabel("Number of Records")
plt.tight_layout()
plt.savefig("outputs/top_crops.png", dpi=150)
plt.close()

# 3. Average yield by season
season_yield = df.groupby("Season")["Yield"].mean().sort_values()

plt.figure(figsize=(9, 5))
season_yield.plot(kind="bar")
plt.title("Average Yield by Season")
plt.ylabel("Average Yield")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("outputs/yield_by_season.png", dpi=150)
plt.close()

# 4. Correlation heatmap for numeric features
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Numeric Feature Correlation")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.close()

print("\nEDA completed successfully!")
print("Charts saved in the outputs folder.")
