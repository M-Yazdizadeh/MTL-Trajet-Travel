import geopandas as gpd

# Step 1: Load the GeoJSON File
def load_geojson(file_path):
    """
    Load a GeoJSON file into a GeoDataFrame.
    This function reads a GeoJSON file using GeoPandas' `read_file` function
    and returns the data as a GeoDataFrame. If the file cannot be loaded, it
    prints an error message and returns None.
    """
    try:
        # Attempt to load the GeoJSON file into a GeoDataFrame
        data = gpd.read_file(file_path)
        # Print the number of records loaded successfully
        print(f"File loaded successfully with {len(data)} records.")
        return data
    except Exception as e:
        # In case of error (e.g., file not found), print the error
        print(f"Error loading GeoJSON file: {e}")
        return None


# Step 2: Inspect the Data
def inspect_data(data):
    """
    Display basic information about the GeoDataFrame.
    This function provides an overview of the data structure, including its
    general properties, geometry types, coordinate reference system (CRS),
    and spatial extent (bounding box).
    """
    print("Data Info:")
    # Display general information about the data (column names, types, non-null counts)
    print(data.info())

    print("\nSample Data:")
    # Show the first few rows of the dataset to get a quick glance at the data
    print(data.head())

    print("\nGeometry Types:")
    # Show the counts of each geometry type (e.g., Point, Polygon, LineString)
    print(data.geom_type.value_counts())

    print("\nCoordinate Reference System (CRS):")
    # Display the Coordinate Reference System (CRS) used in the data
    print(data.crs)

    print("\nSpatial Extent:")
    # Display the spatial extent (bounding box) of the dataset, which is the min and max coordinates
    print(data.total_bounds)


# Step 3: Clean the Data
def clean_data(data):
    """
    Clean the GeoDataFrame by handling missing values, filtering geometries,
    and removing invalid rows.
    This function removes rows with missing or invalid geometries, and fills
    missing attribute values with a default value ("Unknown" for string columns).
    """
    print("\nCleaning data...")
    # Print the initial number of records
    print(f"Initial records: {len(data)}")

    # Drop rows where the 'geometry' column is missing (NaN values)
    data = data.dropna(subset=["geometry"])
    print(f"After dropping missing geometries: {len(data)} records.")

    # Remove rows where the geometries are invalid (e.g., self-intersections)
    data = data[data.is_valid]
    print(f"After removing invalid geometries: {len(data)} records.")

    # Fill missing values in string/object columns with the default value 'Unknown'
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col].fillna("Unknown", inplace=True)
    print("Missing attribute values filled.")

    # Print the number of records after cleaning
    print(f"Cleaned data now contains {len(data)} records.")
    return data


# Step 4: Preprocess the Data
def preprocess_data(data):
    """
    Preprocess the GeoDataFrame.
    This function is a placeholder for any additional preprocessing steps, such
    as spatial transformations, feature engineering, etc.
    Currently, it simply returns the data unchanged.
    """
    print("Preprocessing complete.")
    return data


# Step 5: Save the Processed Data
def save_processed_data(data, output_path):
    """
    Save the processed GeoDataFrame to a GeoJSON file.
    This function attempts to save the cleaned and processed data to a GeoJSON file.
    If the data is empty, it prints a warning message and doesn't save the file.
    """
    print("\nSaving processed data...")
    
    # Check if the GeoDataFrame is empty. If it is, display a warning message.
    if data.empty:
        print("Warning: GeoDataFrame is empty. No data will be saved.")
        return
    
    try:
        # Save the cleaned and processed data to the specified file path
        data.to_file(output_path, driver="GeoJSON")
        print(f"Processed data saved to {output_path}.")
    except Exception as e:
        # If an error occurs during saving, print the error message
        print(f"Error saving processed data: {e}")


# Main Workflow
if __name__ == "__main__":
    # Input file path (the path to the GeoJSON file you want to load)
    input_file = "points_mtl_trajet_2017-1.geojson"

    # Output file path (where you want to save the processed data)
    output_file = "processed_data_points.geojson"

    # Load the GeoJSON file into a GeoDataFrame
    gdf = load_geojson(input_file)

    # If the file is loaded successfully, proceed with inspecting, cleaning, preprocessing, and saving the data
    if gdf is not None:
        # Inspect the data to understand its structure and properties
        inspect_data(gdf)

        # Clean the data (remove invalid geometries, handle missing values)
        gdf_cleaned = clean_data(gdf)

        # Preprocess the data (currently no preprocessing is done, but this step is a placeholder)
        gdf_preprocessed = preprocess_data(gdf_cleaned)

        # Save the processed data to a new GeoJSON file
        save_processed_data(gdf_preprocessed, output_file)
