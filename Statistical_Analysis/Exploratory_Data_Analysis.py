import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from datetime import datetime

# Load the dataset
file_path = 'first_10000_rows.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Step 1: Data Cleaning & Preprocessing

# Convert datetime columns to pandas datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['starttime'] = pd.to_datetime(df['starttime'])
df['endtime'] = pd.to_datetime(df['endtime'])

# Handle missing values

df.dropna(subset=['altitude'], inplace=True)  # Drop rows with missing altitude


# Step 2: Descriptive Statistics

# Display summary statistics for numerical columns
print("Summary Statistics:")
print(df.describe())

# Display counts for categorical columns
print("\nCategorical Features Summary:")
print(df['mode'].value_counts())
print(df['purpose'].value_counts())

# Step 3: Visualizing Key Features

# Visualize the geographical distribution (latitude vs longitude)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='longitude', y='latitude', data=df, alpha=0.6, s=10)
plt.title('Geographical Distribution of Trips')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Visualize the speed distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['speed'], kde=True, color='blue')
plt.title('Speed Distribution of Trips')
plt.xlabel('Speed (km/h)')
plt.ylabel('Frequency')
plt.show()

# Step 4: Time Analysis

# Plot trips over time: Analyze trips by hour of the day
df['hour'] = df['timestamp'].dt.hour
plt.figure(figsize=(8, 6))
sns.countplot(x='hour', data=df, palette='viridis')
plt.title('Trips by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.show()

# Trip duration: Calculate duration of each trip (in minutes)
df['trip_duration'] = (df['endtime'] - df['starttime']).dt.total_seconds() / 60
plt.figure(figsize=(8, 6))
sns.histplot(df['trip_duration'], kde=True, color='green')
plt.title('Trip Duration Distribution')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.show()


# Step 5: Correlation Matrix
# Compute and visualize correlations between numeric variables
corr_matrix = df[['speed', 'altitude', 'trip_duration']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
