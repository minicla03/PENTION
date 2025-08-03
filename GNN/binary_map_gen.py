import osmnx as ox
from shapely.geometry import box
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm  
import os

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_binary_map(city: str = "Benevento, Italy", grid_size: int = 300) -> tuple[np.ndarray, dict]:
    """
    Generates a binary grid map covering the entire urban area of a city based on building footprints.

    Args:
        city (str): City name. Default is "Benevento, Italy".
        grid_size (int): Number of cells per side for the grid.

    Returns:
        np.ndarray: 2D binary map (1: free space, 0: building)
    """
    logging.info(f"Generating binary map for {city} (full extent)...")
    try:
        buildings = ox.features_from_place(city, tags={"building": True})
        if buildings.empty:
            logging.warning(f"No building data found for {city}. Returning an empty map.")
            return np.zeros((grid_size, grid_size), dtype=np.uint8), {}
        
        logging.info(f"Found {len(buildings)} building features.")
        # Get urban boundary polygon
        city_gdf = ox.geocode_to_gdf(city)

    except Exception as e:
        logging.error(f"Error retrieving building data for {city}: {e}")
        return np.zeros((grid_size, grid_size), dtype=np.uint8), {}

    # Project buildings to UTM 33N
    projected_buildings_crs = 32633
    buildings_proj = buildings.to_crs(epsg=projected_buildings_crs)
    city_gdf_proj = city_gdf.to_crs(epsg=projected_buildings_crs)
    urban_polygon_geom = city_gdf_proj.geometry.iloc[0]
    logging.info(f"Buildings projected to EPSG:{projected_buildings_crs}.")

    # Get full bounding box of all buildings
    x_min, y_min, x_max, y_max = urban_polygon_geom.bounds
    logging.info(f"Total bounds: xmin={x_min:.1f}, ymin={y_min:.1f}, xmax={x_max:.1f}, ymax={y_max:.1f}")

    # Compute grid cell size
    cell_width = (x_max - x_min) / grid_size
    cell_height = (y_max - y_min) / grid_size

     # Metadata per riferimenti futuri
    metadata = {
        'city': city,
        'grid_size': grid_size,
        'bounds': (x_min, y_min, x_max, y_max),
        'cell_size': (cell_width, cell_height),
        'crs': projected_buildings_crs,
        'total_buildings': len(buildings_proj)
    }

    binary_grid = np.ones((grid_size, grid_size), dtype=np.uint8)

    logging.info(f"Creating {grid_size}x{grid_size} grid over entire city area.")

    # Ottimizzazione: crea spatial index per ricerche più veloci
    buildings_sindex = buildings_proj.sindex

    # Fill binary grid
    for i in tqdm(range(grid_size), desc="Processing grid"):
        for j in range(grid_size):
            cell = box(
                x_min + i * cell_width,
                y_min + j * cell_height,
                x_min + (i + 1) * cell_width,
                y_min + (j + 1) * cell_height,
            )

            # Usa spatial index per ricerca più efficiente
            possible_matches_index = list(buildings_sindex.intersection(cell.bounds))
            possible_matches = buildings_proj.iloc[possible_matches_index]

            # Controlla intersezioni precise
            if not possible_matches.empty and possible_matches.intersects(cell).any():
                binary_grid[j, i] = 0

    # Calcola statistiche
    total_cells = grid_size * grid_size
    building_cells = np.sum(binary_grid == 0)
    free_cells = np.sum(binary_grid == 1)
    
    logging.info("Binary map generation complete.")
    logging.info(f"Statistics: {building_cells}/{total_cells} cells with buildings ({building_cells/total_cells*100:.1f}%)")
    
    return binary_grid, metadata


if __name__ == "__main__":

    target_city = "Apricena, Italy"
    output_filename = os.path.join(".", "GNN", f"{target_city.lower().replace(', ', '_').replace(' ', '_')}_full_map.npy")
    metadata_filename = os.path.join(".", "GNN", f"{target_city.lower().replace(', ', '_').replace(' ', '_')}_metadata.npy")
    
    binary_map, metadata = generate_binary_map(city=target_city, grid_size=300)

    if binary_map is not None and binary_map.size > 0:

        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        np.save(output_filename, binary_map)
        np.save(metadata_filename, metadata)

        logging.info(f"Binary map saved to '{output_filename}'. Shape: {binary_map.shape}")
        
        plt.figure(figsize=(8, 8))
        plt.imshow(binary_map, cmap="Greys", origin="lower")

        building_cells = np.sum(binary_map == 0)
        free_cells = np.sum(binary_map == 1)

        info_text = f"""
            Informazioni Mappa:
            • Griglia: {metadata.get('grid_size', 'N/A')}×{metadata.get('grid_size', 'N/A')}
            • Edifici totali: {metadata.get('total_buildings', 'N/A')}
            • Celle edifici: {building_cells:,}
            • Celle libere: {free_cells:,}
            • CRS: EPSG:{metadata.get('crs', 'N/A')}
            """
    
        plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.title("Mappa binaria completa di Benevento (0 = edificio, 1 = suolo libero)")
        plt.xlabel("Coordinate X (grid)")
        plt.ylabel("Coordinate Y (grid)")
        plt.colorbar(label="Occupazione")
        plt.grid(False)
        plt.show()
    else:
        logging.error("Binary map was not generated successfully.")
