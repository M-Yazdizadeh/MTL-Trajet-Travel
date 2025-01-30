import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a directory to save plots if it doesn't exist
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
file_path = 'first_10000_rows.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Mode translation dictionary
mode_translation = {
    'Voiture / Moto': 'Car / Motorbike',
    'Transport collectif': 'Public Transport',
    'Vélo': 'Bicycle',
    'À pied': 'Walking',
    'À pied, Transport collectif': 'Walking + Public Transport',
    'À pied, Transport collectif, Vélo': 'Walking + Public Transport + Bicycle',
    'À pied, Voiture / Moto, Vélo': 'Walking + Car / Motorbike + Bicycle',
    'À pied, Vélo': 'Walking + Bicycle',
    'Transport collectif, Voiture / Moto': 'Public Transport + Car / Motorbike'
}

# Purpose translation dictionary
purpose_translation = {
    'Reconduire / aller chercher une personne': 'Dropping off / Picking up someone',
    "Travail / Rendez-vous d'affaires": 'Work / Business appointment',
    'Retourner à mon domicile': 'Returning to my home',
    'Santé': 'Health',
    'Loisir': 'Leisure',
    'Magasinage / emplettes': 'Shopping / Errands',
    'Éducation': 'Education'
}

# Translate modes
df['mode_translated'] = df['mode'].map(mode_translation)

# Translate purposes (handle missing values)
df['purpose_translated'] = df['purpose'].map(purpose_translation).fillna('Other')

# Convert timestamps to datetime
df['starttime'] = pd.to_datetime(df['starttime'])
df['endtime'] = pd.to_datetime(df['endtime'])

# Calculate trip duration (in minutes)
df['duration'] = (df['endtime'] - df['starttime']).dt.total_seconds() / 60

# Assuming speed is in km/h, calculate distance traveled for each trip
df['distance'] = df['speed'] * (df['duration'] / 60)

# ------------------------------
# 1. Extract Unique Trips and Start/End Points
# ------------------------------

# Extract start and end points by finding the first and last point for each trip
start_points = df.groupby('id_trip').first()[['latitude', 'longitude']].rename(columns={'latitude': 'start_lat', 'longitude': 'start_lon'})
end_points = df.groupby('id_trip').last()[['latitude', 'longitude']].rename(columns={'latitude': 'end_lat', 'longitude': 'end_lon'})

# Create a unique trips dataframe
unique_trips = df.drop_duplicates(subset='id_trip')[['id_trip', 'starttime', 'endtime', 'purpose', 'mode', 'mode_translated', 'purpose_translated']]
unique_trips = unique_trips.merge(start_points, left_on='id_trip', right_index=True)
unique_trips = unique_trips.merge(end_points, left_on='id_trip', right_index=True)

# Add trip duration to the unique trips dataframe
unique_trips['starttime'] = pd.to_datetime(unique_trips['starttime'])
unique_trips['endtime'] = pd.to_datetime(unique_trips['endtime'])
unique_trips['duration_minutes'] = (unique_trips['endtime'] - unique_trips['starttime']).dt.total_seconds() / 60

# ------------------------------
# 2. Frequency Analysis
# ------------------------------

# Count trips by translated mode
mode_counts = unique_trips['mode_translated'].value_counts()

# Plot mode distribution and save it
plt.figure(figsize=(10, 6))
sns.barplot(x=mode_counts.index, y=mode_counts.values, palette='viridis')
plt.title('Frequency of Trips by Mode of Transport')
plt.xlabel('Mode of Transport')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mode_distribution.png'))

# Count trips by translated purpose
purpose_counts = unique_trips['purpose_translated'].value_counts()

# Plot purpose distribution and save it
plt.figure(figsize=(10, 6))
sns.barplot(x=purpose_counts.index, y=purpose_counts.values, palette='coolwarm')
plt.title('Frequency of Trips by Purpose')
plt.xlabel('Trip Purpose')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'purpose_distribution.png'))

# ------------------------------
# 3. Average Speed and Total Distance by Mode
# ------------------------------

# Calculate average speed and total distance for each translated mode
avg_speed_per_mode = df.groupby('mode_translated')['speed'].mean()


# Plot average speed by mode and save it
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_speed_per_mode.index, y=avg_speed_per_mode.values, palette='Blues')
plt.title('Average Speed by Mode of Transport')
plt.xlabel('Mode of Transport')
plt.ylabel('Average Speed (km/h)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'average_speed_by_mode.png'))


# ------------------------------
# 4. Boxplots for Speed and Distance
# ------------------------------

# Boxplot for speed by mode and save it
plt.figure(figsize=(12, 8))
sns.boxplot(x='mode_translated', y='speed', data=df, palette='Set2')
plt.title('Distribution of Speed by Mode of Transport')
plt.xlabel('Mode of Transport')
plt.ylabel('Speed (km/h)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'speed_distribution_by_mode.png'))

# Boxplot for distance by mode and save it
plt.figure(figsize=(12, 8))
sns.boxplot(x='mode_translated', y='distance', data=df, palette='Set3')
plt.title('Distribution of Distance by Mode of Transport')
plt.xlabel('Mode of Transport')
plt.ylabel('Distance (km)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'distance_distribution_by_mode.png'))

