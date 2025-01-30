import geopandas as gpd

# Step 1: Load the GeoJSON file
file_path = "merged_file.geojson"
# Use GeoPandas to read the GeoJSON file into a GeoDataFrame
gdf = gpd.read_file(file_path)

# Step 2: Inspect the data to confirm column names
# Print the column names of the dataset to verify the structure of the data
print("Columns in the dataset:", gdf.columns)

# Step 3: Check for missing or empty values in the 'mode' column
# Print the total number of rows in the GeoDataFrame
print(f"Total rows: {len(gdf)}")
# Count the number of missing values (NaN) in the 'mode' column
missing_mode_count = gdf['mode'].isna().sum()
print(f"Number of missing values in 'mode': {missing_mode_count}")

# Step 4: Drop rows where 'mode' column is missing (NaN)
# Drop the rows where the 'mode' column contains NaN (missing) values
filtered_gdf = gdf.dropna(subset=['mode'])

# Step 5: Handle empty strings, whitespace, and the value "unknown" in 'mode'
# Check if the 'mode' column is of type 'object' (string type)
if filtered_gdf['mode'].dtype == 'object':
    # Count the number of empty strings or whitespace values in the 'mode' column
    empty_mode_count = filtered_gdf['mode'].str.strip().eq("").sum()
    print(f"Number of empty or whitespace 'mode' values: {empty_mode_count}")
    # Remove rows where the 'mode' value is empty or contains only whitespace
    filtered_gdf = filtered_gdf[filtered_gdf['mode'].str.strip() != ""]
    
    # Count the number of rows where 'mode' is "unknown" (case-insensitive)
    unknown_mode_count = (filtered_gdf['mode'].str.strip().str.lower() == "unknown").sum()
    print(f"Number of rows with 'unknown' in 'mode': {unknown_mode_count}")
    # Remove rows where the 'mode' value is "unknown" (case-insensitive)
    filtered_gdf = filtered_gdf[filtered_gdf['mode'].str.strip().str.lower() != "unknown"]

# Step 6: Save the cleaned GeoDataFrame to a new GeoJSON file
# Define the output path for the cleaned GeoDataFrame (in GeoJSON format)
output_path_geojson = "filtered_file.geojson"
# Save the cleaned GeoDataFrame to a new GeoJSON file using the 'to_file' method
filtered_gdf.to_file(output_path_geojson, driver="GeoJSON")
# Print confirmation of the saved file path
print(f"Cleaned data saved to: {output_path_geojson}")

# Step 7: Save the first 100000 rows to a CSV file
# Define the output path for the CSV file
output_path_csv = "first_100000_rows.csv"
# Save the first 100000 rows of the filtered GeoDataFrame to a CSV file
filtered_gdf.head(100000).to_csv(output_path_csv, index=False)
# Print confirmation of the saved CSV file path
print(f"The first 100000 rows saved to: {output_path_csv}")
