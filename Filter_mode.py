import geopandas as gpd

# Load the GeoJSON file
file_path = "merged_file.geojson"
gdf = gpd.read_file(file_path)

# Inspect the data to confirm column names
print("Columns in the dataset:", gdf.columns)

# Check for missing or empty values in the 'mode' column
print(f"Total rows: {len(gdf)}")
missing_mode_count = gdf['mode'].isna().sum()
print(f"Number of missing values in 'mode': {missing_mode_count}")

# Drop rows where 'mode' column is missing (NaN)
filtered_gdf = gdf.dropna(subset=['mode'])

# Handle empty strings, whitespace, and the value "unknown" in 'mode'
if filtered_gdf['mode'].dtype == 'object':
    empty_mode_count = filtered_gdf['mode'].str.strip().eq("").sum()
    print(f"Number of empty or whitespace 'mode' values: {empty_mode_count}")
    filtered_gdf = filtered_gdf[filtered_gdf['mode'].str.strip() != ""]
    
    # Remove rows where 'mode' is "unknown"
    unknown_mode_count = (filtered_gdf['mode'].str.strip().str.lower() == "unknown").sum()
    print(f"Number of rows with 'unknown' in 'mode': {unknown_mode_count}")
    filtered_gdf = filtered_gdf[filtered_gdf['mode'].str.strip().str.lower() != "unknown"]

# Save the cleaned GeoDataFrame to a new GeoJSON file
output_path_geojson = "filtered_file.geojson"
filtered_gdf.to_file(output_path_geojson, driver="GeoJSON")
print(f"Cleaned data saved to: {output_path_geojson}")

# Save the first 100000 rows to a CSV file
output_path_csv = "first_100000_rows.csv"
filtered_gdf.head(100000).to_csv(output_path_csv, index=False)
print(f"The first 100000 rows saved to: {output_path_csv}")




