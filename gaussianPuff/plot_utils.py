# plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import matplotlib.animation as animation

def plot_plan_view(C1, x, y, title):
    plt.figure(figsize=(8, 6))
    data = np.mean(C1, axis=2) * 1e6

    vmin = np.percentile(data, 5)
    vmax = np.percentile(data, 95)

    plt.pcolor(x, y, data, cmap='jet')
    plt.clim(vmin, vmax) 
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    cb = plt.colorbar()
    cb.set_label(r'$\mu g \cdot m^{-3}$')
    plt.tight_layout()
    plt.show()

def animate_plan_view(C1, x, y, binary_map=None, sensor_locs=None, interval=200, save_path=None):
    """
    Anima la dispersione temporale su mappa planare.
    
    Parametri:
    - C1: array (Y, X, T)
    - x, y: meshgrid (2D) delle coordinate in metri
    - binary_map: (Y, X) - 1 = suolo libero, 0 = edificio
    - sensor_locs: lista di tuple (x, y) in metri
    - interval: tempo tra i frame in ms
    - save_path: se fornito, salva la gif (es. 'dispersion.gif')
    """
    assert C1.ndim == 3, "C1 deve avere shape (Y, X, T)"
    Y, X, T = C1.shape

    # Conversione a microgrammi/m^3
    C1_micro = C1 * 1e6

    vmin = np.percentile(C1_micro, 5)
    vmax = np.percentile(C1_micro, 95)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('jet')

    # Primo frame
    img = ax.pcolormesh(x, y, C1_micro[:, :, 0], cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(img, ax=ax)
    cb.set_label(r'$\mu g \cdot m^{-3}$')

    # Overlay edifici
    if binary_map is not None:
        buildings_overlay = ax.imshow((binary_map == 0), extent=(x.min(), x.max(), y.min(), y.max()),
                                      origin='lower', cmap='Greys', alpha=0.3)

    # Overlay sensori
    if sensor_locs is not None:
        sensor_scatter = ax.scatter(*zip(*sensor_locs), marker='^', c='black', s=80, label='Sensori')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    title = ax.set_title("Dispersione al tempo t = 0")

    def update(t):
        img.set_array(C1_micro[:, :, t].ravel())
        title.set_text(f"Dispersione al tempo t = {t}")
        return img, title

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=False)

    plt.tight_layout()

    if save_path:
        ani.save(save_path, writer='pillow', fps=1000//interval)
        print(f"✅ Animazione salvata in: {save_path}")
    else:
        plt.show()

def plot_plan_view_2(C_mean, x, y, title):
    plt.figure(figsize=(8, 6))

    vmin = np.percentile(C_mean, 5)
    vmax = np.percentile(C_mean, 95)

    plt.pcolor(x, y, C_mean*1e6, cmap='jet')
    plt.clim(vmin, vmax)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    cb = plt.colorbar()
    cb.set_label(r'$\mu g \cdot m^{-3}$')
    plt.tight_layout()
    plt.show()

def plot_surface_time(C1, times, x_idx, y_idx, stability, stab_label, wind_label):
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 6))
    signal = 1e6 * np.squeeze(C1[y_idx, x_idx, :])
    ax1.plot(times, signal, label="Hourly mean")
    ax1.plot(times, smooth(signal, 24), 'r', label="Daily mean")
    ax1.set_ylabel('Mass loading ($m$ g m$^{-3}$)')
    ax1.set_title(stab_label + '\n' + wind_label)
    ax1.legend()

    ax2.plot(times, stability)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Stability')

    plt.tight_layout()
    plt.show()

def plot_height_slice(C1, y, z, stab_label, wind_label):
    plt.figure(figsize=(8, 6))
    data = np.mean(C1, axis=2) * 1e6
    plt.pcolor(y,z, data, cmap='jet')      
    plt.clim(0,1e2)
    plt.xlabel('y (metres)')
    plt.ylabel('z (metres)')
    plt.title(stab_label + '\n' + wind_label)
    cb1=plt.colorbar()
    cb1.set_label(r'$\mu$ g m$^{-3}$')
    plt.show()

def plot_puff_on_map(C1, x_grid, y_grid, center_lat, center_lon, timestep=-1, threshold=0.00, cutoff_norm=0.10, zoom_start=13, sensor_locs=None):
    deg_per_m = 1 / 111320

    lat_grid = center_lat + y_grid * deg_per_m
    lon_grid = center_lon + x_grid * deg_per_m / np.cos(np.deg2rad(center_lat))

    C = C1[:, :, timestep] if timestep >= 0 else np.mean(C1, axis=2)
    C_max = np.max(C)
    print(f"Max concentrazione: {C_max}")
    if C_max == 0:
        raise ValueError("Tutte le concentrazioni sono nulle")

    C_norm = C / C_max

    print(f"Valori normalizzati: min {np.min(C_norm)}, max {np.max(C_norm)}")

    points = []
    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            if C[i, j] > threshold and C_norm[i, j] > cutoff_norm:
                points.append([lat_grid[i, j], lon_grid[i, j], C_norm[i, j]])
    print(f"Punti selezionati: {len(points)}")

    if not points:
        raise ValueError("Nessuna concentrazione supera la soglia impostata")

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="cartodbpositron")
    HeatMap(points, radius=10, blur=5, max_zoom=1).add_to(m)

    folium.CircleMarker(
        location=[center_lat, center_lon],
        radius=7,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.9,
        popup='Punto di origine'
    ).add_to(m)

    legend_html = '''
     <div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white;
     opacity: 0.8;
     padding: 10px;
     ">
     <b>Legenda concentrazione</b><br>
     <i style="background: linear-gradient(to right, blue, red); 
         display:inline-block; width: 100px; height: 10px;"></i><br>
     <small>Blu: basso</small><br>
     <small>Rosso: alto</small>
     </div>
     '''
    m.get_root().add_child(folium.Element(legend_html))
    if sensor_locs:
        m = plot_sensors_on_map(sensor_locs, m)
    return m

def meters_to_latlon(x, y, origin_lat, origin_lon):
    R = 6378137  # raggio terrestre medio (m)
    dLat = y / R
    dLon = x / (R * np.cos(np.pi * origin_lat / 180))
    lat = origin_lat + dLat * (180 / np.pi)
    lon = origin_lon + dLon * (180 / np.pi)
    return lat, lon

def folium_map_plot(sensor_coords_geo, source_geo, map_center=None, zoom_start=14):
    if map_center is None:
        map_center = np.mean(sensor_coords_geo, axis=0)

    m = folium.Map(location=map_center, zoom_start=zoom_start)

    # Sensori
    for i, (lat, lon) in enumerate(sensor_coords_geo):
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="red", icon="info-sign"),
            popup=f"Sensore {i+1}"
        ).add_to(m)

    # Sorgente stimata
    folium.Marker(
        location=source_geo,
        icon=folium.Icon(color="green", icon="star"),
        popup="Sorgente stimata"
    ).add_to(m)

    return m

def plot_sensors_on_map(sensor_positions, mappa):
    import folium
    for pos in sensor_positions:
        folium.Marker(location=pos, popup="Sensore", icon=folium.Icon(color='blue')).add_to(mappa)
    for i, pos in enumerate(sensor_positions):
        folium.Marker(location=pos, popup=f"Sensore {i+1}", icon=folium.Icon(color='blue')).add_to(mappa)
    return mappa

def plot_plan_view_with_mask(C1, x, y, binary_map, sensor_locs=None, title="", save_path=None, show=True):
    """
    Plotta la media temporale della concentrazione con sovrapposizione della mappa binaria di Benevento.

    - C1: array (Y, X, T)
    - x, y: meshgrid delle coordinate
    - binary_map: array (Y, X), 1 = suolo libero, 0 = edificio
    - sensor_locs: lista opzionale di tuple (x, y)
    - title: titolo del grafico
    - save_path: percorso file per salvare l'immagine (es. "mappa.png"), se None non salva
    - show: se True mostra la figura con plt.show()
    
    Ritorna:
    - fig, ax: figure e axis matplotlib per eventuali modifiche successive
    """
    assert C1.ndim == 3, "C1 deve essere 3D (Y, X, T)"
    assert binary_map.shape == C1.shape[:2], "binary_map deve avere shape compatibile con C1"

    # Media temporale e conversione in μg/m³
    data = np.mean(C1, axis=2) * 1e6

    vmin = np.percentile(data, 5)
    vmax = np.percentile(data, 95)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Mappa concentrazione
    c = ax.pcolor(x, y, data, cmap='jet', shading='auto')
    c.set_clim(vmin, vmax)

    # Overlay edifici (dove binary_map == 0)
    ax.imshow((binary_map == 0), extent=(x.min(), x.max(), y.min(), y.max()),
              origin='lower', cmap='Greys', alpha=0.3)

    # Sensori opzionali
    if sensor_locs:
        sx, sy = zip(*sensor_locs)
        ax.scatter(sx, sy, c='black', marker='^', s=100, label='Sensori')
        ax.legend()

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)

    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r'$\mu g \cdot m^{-3}$')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"✅ Figura salvata in: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax

def plot_concentration_with_sensors(C, x, y, sensors, source, times, time_index=0, title=""):
    """
    C: array (X, Y, T)
    x, y: assi griglia
    sensors: lista di tuple (x, y)
    source: tuple (x_src, y_src)
    """
    conc_slice = C[:, :, time_index]

    extent = (x.min(), x.max(), y.min(), y.max())
    fig, ax = plt.subplots(figsize=(8, 6))
    im=ax.imshow(conc_slice.T, origin='lower', extent=extent, cmap='viridis')

    plt.colorbar(im, ax=ax, label="Concentrazione (a.u.)")

    # Sensori
    sensors_x, sensors_y = zip(*sensors)
    ax.scatter(sensors_x, sensors_y, c="red", marker="o", s=50, label="Sensori")

    # Sorgente
    ax.scatter(source[0], source[1], c="black", marker="*", s=100, label="Sorgente")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or f"Concentrazione al tempo t={round(times[time_index],2)}")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_timeseries(times, conc, sensor_id=None):
    plt.figure(figsize=(8,4))
    plt.plot(times, conc, marker='o')
    plt.xlabel("Tempo")
    plt.ylabel("Concentrazione")
    title = f"Profilo temporale sensore {sensor_id}" if sensor_id is not None else "Profilo temporale"
    plt.title(title)
    plt.grid()
    plt.show()
