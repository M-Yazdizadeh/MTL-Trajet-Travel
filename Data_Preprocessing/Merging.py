import geopandas as gpd

# Step 1: Load the first GeoJSON file
file1_path = "processed_data_points.geojson"
# Using GeoPandas to read the first GeoJSON file into a GeoDataFrame
gdf1 = gpd.read_file(file1_path)

# Step 2: Load the second GeoJSON file
file2_path = "processed_data_trips.geojson"
# Using GeoPandas to read the second GeoJSON file into a GeoDataFrame
gdf2 = gpd.read_file(file2_path)

# Step 3: Merge the GeoDataFrames on the 'id_trip' column
# The 'merge' function combines the two GeoDataFrames based on the common 'id_trip' column.
# It performs an inner join by default, meaning only rows with matching 'id_trip' in both dataframes will be kept.
merged_gdf = gdf1.merge(gdf2, on='id_trip')

# Step 4: Check the geometry columns
# Identify the columns in the merged GeoDataFrame that contain geometry data.
# GeoPandas assigns geometries to columns with the data type 'geometry'.
geometry_columns = [col for col in merged_gdf.columns if merged_gdf[col].dtype.name == 'geometry']
# Print the list of geometry columns
print("Geometry columns:", geometry_columns)

# Step 5: Retain only one geometry column (e.g., from gdf1)
# The 'set_geometry' function sets a specific column as the geometry column.
# We select the first geometry column (from gdf1) to be used as the geometry for the merged GeoDataFrame.
merged_gdf = merged_gdf.set_geometry(geometry_columns[0])

# Step 6: Optionally, drop the other geometry columns
# After selecting one geometry column, we can drop the remaining geometry columns to avoid redundancy.
for col in geometry_columns[1:]:
    merged_gdf = merged_gdf.drop(columns=[col])

# Step 7: Save the merged GeoDataFrame to a new GeoJSON file
# The 'to_file' method saves the merged GeoDataFrame to a new GeoJSON file.
output_path = "merged_file.geojson"
merged_gdf.to_file(output_path, driver='GeoJSON')

# Step 8: Print the success message
# After successfully saving the merged data, we print a message indicating where the file has been saved.
print(f"Merged GeoJSON saved to {output_path}")

