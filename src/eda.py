import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/data/insurance_premium_dataset.csv")

print(df.info())
print(df.describe())

# Distribution
sns.histplot(df["annual_premium"], kde=True)
plt.title("Premium Distribution")
plt.show()

# City-wise premium
plt.figure(figsize=(10,5))
sns.boxplot(x="city", y="annual_premium", data=df)
plt.xticks(rotation=30)
plt.show()

# Correlation
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="coolwarm")
plt.show()

