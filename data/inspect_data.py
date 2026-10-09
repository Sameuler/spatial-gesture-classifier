import pandas as pd

df = pd.read_csv("gestures.csv")
print(df["label"].value_counts())
print(df.shape)
print(df.isna().sum().sum())
print(df.describe())