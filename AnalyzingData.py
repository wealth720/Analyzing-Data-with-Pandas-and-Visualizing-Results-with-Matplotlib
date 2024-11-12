# Task 1: Load and Explore the Dataset

import pandas as pd
from sklearn.datasets import load_iris

# Load the dataset and convert it to a DataFrame
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nDataset Info:")
print(df.info())

# Check for any missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Clean the dataset if needed
# In this case, the Iris dataset has no missing values, so no cleaning is necessary

# Step 2: Explore the Structure of the Dataset

# Check the data types of each column
print("\nData Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Clean the Dataset (if necessary)

# Fill missing values (if any) with the mean of the column
df.fillna(df.mean(), inplace=True)

# Alternatively, you can drop rows with missing values
# df.dropna(inplace=True)

# Task 2: Basic Data Analysis

# Basic statistics of numerical columns
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute the mean for each group
species_means = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(species_means)


# Task 3: Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns

# Set the Seaborn style
sns.set(style="whitegrid")

# Line Chart
plt.figure(figsize=(10, 5))
sns.lineplot(data=species_means, markers=True, dashes=False)
plt.title("Average Measurements of Iris Species")
plt.xlabel("Species")
plt.ylabel("Measurement (cm)")
plt.legend(species_means.columns, loc='upper right')
plt.show()

# Bar Chart
plt.figure(figsize=(10, 5))
sns.barplot(x="species", y="petal length (cm)", data=df, ci=None)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df["sepal length (cm)"], kde=True, bins=10)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df)
plt.title("Sepal Length vs. Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# Additional Instructions for Error Handling

try:
    # Attempt to load a CSV file if provided
    df = pd.read_csv('path_to_your_dataset.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("File not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("The file is empty.")
except Exception as e:
    print(f"An error occurred: {e}")
