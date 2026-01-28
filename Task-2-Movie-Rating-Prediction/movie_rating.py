import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Load dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
# Show first 5 rows
print(df.head())
# Dataset info
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Select important columns
df = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Votes', 'Rating']]

# Drop rows with missing values
df = df.dropna()

print("\nAfter cleaning shape:", df.shape)
