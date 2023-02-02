import folium
import pandas as pd
# Load the data into a pandas DataFrame
# df = pd.read_csv("data.csv")
# Create a map centered on the average latitude and longitude


def generate_map(train: pd.DataFrame) -> folium.Map:
    avg_lat = train['latitude'].mean()
    avg_lon = train['longitude'].mean()
    map = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)
    # Add markers for each location in the DataFrame
    train.apply(
        lambda row: folium.Marker(
            [row['latitude'], row['longitude']]).add_to(map),
        axis=1)
    # Display the map
    return map
