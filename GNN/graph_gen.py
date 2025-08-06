import torch
import numpy as np
import os
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
from sklearn.preprocessing import LabelEncoder

def create_graph(datset_copy, binary_map, t_idx=300, sensor=False):
    graph_list=[]
    if sensor:
        for idx in tqdm(range(0, len(datset_copy), 5), desc="Creating graphs from sensors"):
            graph = create_graph_from_sensors(datset_copy.iloc[idx:idx+5], binary_map)
            graph_list.append(graph)
    else:
        for idx, row in tqdm(datset_copy.iterrows(), total=len(datset_copy), desc="Creating graphs from simulations"):
            graph=create_graph_from_simulation(row, binary_map, t_idx)
            graph_list.append(graph)
    return graph_list

def create_graph_from_sensors(rows, binary_map, t_idx=300):
    """
    rows: DataFrame con 5 righe corrispondenti a una simulazione
    """
    assert len(rows) == 5, "Ogni grafo deve essere costruito da 5 righe (5 sensori)"
    
    H, W = binary_map.shape
    num_nodes = len(rows)

    nodes   = []
    node_features = []
    sensor_concentrations = []
    pos_list = []
    conc_targets = []

    # Prepara encoder per le stringhe
    le_wind_type = LabelEncoder()
    le_aerosol = LabelEncoder()
    le_stability = LabelEncoder()

    # Fit encoder su tutto il dataset una volta
    le_wind_type.fit(rows['wind_type'])
    le_aerosol.fit(rows['aerosol_type'])
    le_stability.fit(rows['stability_value'])

    print(f"Creating graph with {num_nodes} nodes from {len(rows)} sensors")

    #nodo della sorgente
    row_data = rows.iloc[0]
    source_features = [
        row_data['source_x'], row_data['source_y'], row_data['source_h'], row_data['emission_rate'], 
        row_data['wind_speed'],  row_data['wind_dir_cos'], row_data['wind_dir_sin'], le_wind_type.transform([row_data['wind_type']])[0],
        row_data['RH'],
        le_stability.transform([row_data['stability_value']])[0], 
        le_aerosol.transform([row_data['aerosol_type']])[0],
        1.0, # source indicator
        0.0,  # filler per sensor_noise
    ]

    nodes.append(('source', row_data['source_x'], row_data['source_y']))
    node_features.append(source_features)
    pos_list.append((row_data['source_x'], row_data['source_y']))

    for _, sensor_row in rows.iterrows():
        sensor_features = [
                sensor_row['sensor_x'], sensor_row['sensor_y'], sensor_row['sensor_noise'],
                row_data['wind_speed'], row_data['RH'], row_data['wind_dir_cos'], row_data['wind_dir_sin'],
                row_data['wind_dir_sin'], le_stability.transform([row_data['stability_value']])[0],
                le_aerosol.transform([row_data['aerosol_type']])[0], 
                0.0, # sensor indicator
                0.0, # filler per emission_rate
                0.0 # filler per source_h
            ]
        nodes.append(('sensor', sensor_row['sensor_x'], sensor_row['sensor_y']))
        node_features.append(sensor_features)
        pos_list.append((sensor_row['sensor_x'], sensor_row['sensor_y']))

        # Parse concentration time series - adattato al tuo formato
        """def convert_string_to_array(s):
            if isinstance(s, str):
                s = s.replace('[', '').replace(']', '')
                return np.fromstring(s, sep=',')
            return s
        conc_series = sensor_row["contratio_series"].apply(convert_string_to_array)"""
        # Prendi la concentrazione media come target semplificato
        avg_concentration = np.mean(sensor_row['contratio_series'])
        sensor_concentrations.append(avg_concentration)
       
    # Crea edge_index e edge_features
    edge_index = []
    edge_features = []
    
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            
            # Calcola distanza
            x1, y1 = nodes[i][1], nodes[i][2]
            x2, y2 = nodes[j][1], nodes[j][2]
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Calcola ostruzione da edifici
            building_obstruction = calculate_building_obstruction(H, x1, y1, x2, y2)
            
            # Features addizionali per gli edge
            wind_direction_influence = abs(np.cos(np.arctan2(y2-y1, x2-x1)))
            
            # Aggiungi edge bidirezionale
            edge_index.extend([[i, j], [j, i]])
            edge_attr = [distance, building_obstruction, wind_direction_influence]
            edge_features.extend([edge_attr, edge_attr])
        
    # Converti in tensori
    x = torch.FloatTensor(node_features)
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    edge_attr = torch.FloatTensor(edge_features)
    y = torch.FloatTensor([np.mean(sensor_concentrations)])  # Solo le concentrazioni dei sensori
    
    data =  Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    data.mask = torch.tensor([binary_map[int(y), int(x)] for x, y in pos_list], dtype=torch.float32)
    return data

def create_graph_from_simulation(row, binary_map, conc_array, t_idx=300):
    H, W = binary_map.shape
    num_nodes = H * W

    is_free = binary_map.flatten().astype(np.float32)

    # Coordinate griglia e distanza da sorgente
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    dists = np.sqrt((xx - row['source_x'])**2 + (yy - row['source_y'])**2).flatten().astype(np.float32)

    conc_gauss = conc_array[:, :, t_idx].flatten().astype(np.float32)

    # Feature globali replicate
    wind_cos = np.full(num_nodes, row['wind_dir_cos'], dtype=np.float32)
    wind_sin = np.full(num_nodes, row['wind_dir_sin'], dtype=np.float32)
    wind_speed = np.full(num_nodes, row['wind_speed'], dtype=np.float32)
    emission_rate = np.full(num_nodes, row['emission_rate'], dtype=np.float32)
    source_h = np.full(num_nodes, row['source_h'], dtype=np.float32)

    aerosol_id = hash(row['aerosol_type']) % 10
    stability_id = hash(row['stability_value']) % 10
    aerosol_type = np.full(num_nodes, aerosol_id, dtype=np.float32)
    stability_value = np.full(num_nodes, stability_id, dtype=np.float32)

    node_features = np.stack([
        is_free, dists, conc_gauss,
        wind_cos, wind_sin, wind_speed,
        emission_rate, source_h, aerosol_type, stability_value
    ], axis=1)

    x = torch.tensor(node_features, dtype=torch.float32)
    y = torch.tensor(conc_gauss, dtype=torch.float32).unsqueeze(1)
    pos = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32)
    edge_index, _ = pyg_utils.grid(H, W)

    data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
    data.mask = torch.tensor(is_free, dtype=torch.float32)  
    return data

def calculate_building_obstruction(binary_map, grid_size, x1, y1, x2, y2):
    """
    Calcola la percentuale di edifici lungo la linea tra due punti
    """
    # Discretizza la linea tra i due punti
    num_points = max(int(np.sqrt((x2-x1)**2 + (y2-y1)**2)), 10)
    x_line = np.linspace(x1, y1, num_points)
    y_line = np.linspace(x2, y2, num_points)
    
    # Conta gli edifici lungo la linea
    building_count = 0
    for x, y in zip(x_line, y_line):
        if 0 <= int(x) < grid_size and 0 <= int(y) < grid_size:
            if binary_map[int(y), int(x)] == 0:  # 0 = edificio
                building_count += 1
    
    return building_count / num_points if num_points > 0 else 0

def plot_graph(data, binary_map=None, title="Grafo con dettagli"):
    G = nx.DiGraph()
    
    # Aggiungi nodi con attributi
    for i, node_feat in enumerate(data.x):
        x_pos, y_pos = node_feat[0].item(), node_feat[1].item()
        is_source = node_feat[11].item() == 1.0
        # Altri valori che vuoi mostrare
        emission_rate = node_feat[3].item() if is_source else None
        sensor_noise = node_feat[2].item() if not is_source else None
        
        G.add_node(i, pos=(x_pos, y_pos), is_source=is_source, 
                   emission_rate=emission_rate, sensor_noise=sensor_noise)
    
    # Aggiungi archi con peso
    edge_index = data.edge_index.t().tolist()
    for idx, (src, dst) in enumerate(edge_index):
        dist = data.edge_attr[idx][0].item() if data.edge_attr is not None else 1.0
        obstruction = data.edge_attr[idx][1].item() if data.edge_attr is not None else 0.0
        G.add_edge(src, dst, distance=dist, obstruction=obstruction)
    
    pos = nx.get_node_attributes(G, 'pos')
    is_source = nx.get_node_attributes(G, 'is_source')
    emission_rate = nx.get_node_attributes(G, 'emission_rate')
    sensor_noise = nx.get_node_attributes(G, 'sensor_noise')

    # Colori e forme nodi
    node_colors = ['red' if is_source[n] else 'blue' for n in G.nodes]
    node_shapes = {True: 'o', False: 's'}  # o cerchio per sorgente, s quadrato per sensore
    
    plt.figure(figsize=(10,10))

    if binary_map is not None:
        plt.imshow(binary_map, cmap='gray_r', origin='lower')

    
    # Disegna i nodi con forme separate
    for shape in set(node_shapes.values()):
        nodes_of_shape = [n for n in G.nodes if node_shapes[is_source[n]] == shape]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_shape, 
                               node_color=[node_colors[n] for n in nodes_of_shape],
                               node_shape=shape,
                               node_size=[(emission_rate.get(n, 0)*1000 + 100) if is_source[n] else (sensor_noise.get(n, 0)*500 + 100) for n in nodes_of_shape],
                               alpha=0.8)
    
    # Disegna gli archi colorandoli in base all'ostruzione
    edges = G.edges(data=True)
    edge_colors = [plt.cm.viridis(e[2]['obstruction']) for e in edges]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowsize=15, width=2)

    # Etichette sugli archi: mostra il valore di ostruzione 
    edge_labels = {(e[0], e[1]): f"{e[2]['obstruction']:.2f}" for e in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Etichette nodi con coordinate (arrotondate)
    labels = {n: f"{pos[n][0]:.1f},{pos[n][1]:.1f}" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='Sorgente')
    blue_patch = mpatches.Patch(color='blue', label='Sensore')
    building_patch = mpatches.Patch(color='black', label='Spazio libero')
    free_space_patch = mpatches.Patch(color='white', label='Edificio')
    
    plt.legend(handles=[red_patch, blue_patch, building_patch, free_space_patch], loc='upper right')    
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.show()
