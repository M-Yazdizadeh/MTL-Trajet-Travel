import geopandas as gpd

# Step 1: Load the GeoJSON File
def load_geojson(file_path):
    """
    Load a GeoJSON file into a GeoDataFrame.
    - This function attempts to read a GeoJSON file using GeoPandas' `read_file` function,
      and it returns the data as a GeoDataFrame.
    - If the file cannot be loaded, it prints an error message and returns None.
    """
    try:
        # Using GeoPandas to read the file
        data = gpd.read_file(file_path)
        print(f"File loaded successfully with {len(data)} records.")
        return data
    except Exception as e:
        # Handling any errors that may occur while loading the file
        print(f"Error loading GeoJSON file: {e}")
        return None


# Step 2: Inspect the Data
def inspect_data(data):
    """
    Display basic information about the GeoDataFrame.
    - This function provides an overview of the loaded GeoDataFrame, including its 
      structure, geometry types, CRS, and spatial extent.
    """
    print("Data Info:")
    # Display general information about the data
    print(data.info())

    print("\nSample Data:")
    # Display the first few rows of the data
    print(data.head())

    print("\nGeometry Types:")
    # Show the counts of each type of geometry (e.g., Point, Polygon, etc.)
    print(data.geom_type.value_counts())

    print("\nCoordinate Reference System (CRS):")
    # Display the CRS (spatial reference) used in the data
    print(data.crs)

    print("\nSpatial Extent:")
    # Display the bounds (min/max x and y coordinates) of the spatial data
    print(data.total_bounds)


# Step 3: Clean the Data
def clean_data(data):
    """
    Clean the GeoDataFrame by handling missing values, filtering geometries, and removing invalid rows.
    - This function removes rows with missing geometries, filters invalid geometries, 
      and fills missing attribute values with a default value ("Unknown" for string columns).
    """
    print("\nCleaning data...")
    print(f"Initial records: {len(data)}")

    # Drop rows with missing geometries
    data = data.dropna(subset=["geometry"])
    print(f"After dropping missing geometries: {len(data)} records.")

    # Ensure all geometries are valid
    data = data[data.is_valid]
    print(f"After removing invalid geometries: {len(data)} records.")

    # Fill missing attribute values in categorical columns with "Unknown"
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col].fillna("Unknown", inplace=True)
    print("Missing attribute values filled.")

    # Return the cleaned data
    print(f"Cleaned data now contains {len(data)} records.")
    return data


# Step 4: Preprocess the Data
def preprocess_data(data):
    """
    Preprocess the GeoDataFrame.
    - This function is a placeholder for any additional preprocessing steps like 
      spatial transformations or feature engineering. Currently, it simply returns the data unchanged.
    """
    # You can add more preprocessing steps here if needed
    print("Preprocessing complete.")
    return data


# Step 5: Save the Processed Data
def save_processed_data(data, output_path):
    """
    Save the processed GeoDataFrame to a GeoJSON file.
    - This function attempts to save the cleaned and processed GeoDataFrame to the specified file path.
    - If the GeoDataFrame is empty, a warning is displayed and no file is saved.
    """
    print("\nSaving processed data...")
    
    # Check if the GeoDataFrame is empty
    if data.empty:
        print("Warning: GeoDataFrame is empty. No data will be saved.")
        return
    
    try:
        # Save the data as a GeoJSON file
        data.to_file(output_path, driver="GeoJSON")
        print(f"Processed data saved to {output_path}.")
    except Exception as e:
        # If an error occurs during saving, print the error message
        print(f"Error saving processed data: {e}")


# Main Workflow
if __name__ == "__main__":
    # Define the input and output file paths
    input_file = "trajets_mtl_trajet_2017-1.geojson"
    output_file = "processed_data_trips.geojson"

    # Load the GeoJSON file into a GeoDataFrame
    gdf = load_geojson(input_file)

    # If the file was loaded successfully, proceed with the workflow
    if gdf is not None:
        # Inspect the data to understand its structure and properties
        inspect_data(gdf)

        # Clean the data by removing invalid entries and handling missing values
        gdf_cleaned = clean_data(gdf)

        # Preprocess the data (if necessary)
        gdf_preprocessed = preprocess_data(gdf_cleaned)

        # Save the cleaned and processed data to a new GeoJSON file
        save_processed_data(gdf_preprocessed, output_file)

