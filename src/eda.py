import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/data/insurance_dataset.csv")

print(df.info())
print(df.describe())

plt.figure(figsize=(8,4))
sns.histplot(df["annual_premium"], kde=True)
plt.title("Annual Premium Distribution")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x="city", y="annual_premium", data=df)
plt.xticks(rotation=30)
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm")
plt.show()
