#gauss_model.py
import numpy as np
from scipy.special import erfcinv
from config import ModelConfig, StabilityType, WindType, NPS, nps_properties, OutputType
from gaussianFunction import gauss_func

# Funzione di crescita igroscopica basata su Köhler
def apply_hygroscopic_growth(C1: np.ndarray, RH: float, dry_size: float, nps_type: NPS) -> np.ndarray:
    Mw = 18e-3
    rho_s = nps_properties[nps_type]["rho_s"]
    Ms = nps_properties[nps_type]["Ms"]
    nu = nps_properties[nps_type]["nu"]

    mass=np.pi/6.*rho_s*dry_size**3.
    moles=mass/Ms
        
    nw=RH*nu*moles/(1.-RH)
    mass2=nw*Mw+moles*Ms
    C1_humidified=C1*mass2/mass; 

    return C1_humidified

def run_dispersion_model(config: ModelConfig):
    dxy = 5000 / 299
    dz = 10
    x = np.mgrid[-2500:2500 + dxy:dxy]
    y = x.copy()
    times = np.arange(1, config.days * 24 + 1) / 24.0

    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.zeros_like(x_grid)

    if config.stability_profile == StabilityType.CONSTANT:
        stability = np.full_like(times, config.stability_value.value)
        stability_label = f"Stability {config.stability_value.value}"
    else:
        stability = np.round(2.5 * np.cos(times * 2 * np.pi / 365.) + 3.5)
        stability_label = "Annual cycle"

    if config.output == OutputType.PLAN_VIEW or config.output == OutputType.SURFACE_TIME or config.output == OutputType.NO_PLOT:
        C1=np.zeros((len(x),len(y),len(times))); # array to store data, initialised to be zero

        [x,y]=np.meshgrid(x,y); # x and y defined at all positions on the grid
        z=np.zeros(np.shape(x));    # z is defined to be at ground level.
    elif config.output == OutputType.HEIGHT_SLICE:
        z=np.mgrid[0:500+dz:dz];       # z-grid

        C1=np.zeros((len(y),len(z),len(times))); # array to store data, initialised to be zero

        [y,z]=np.meshgrid(y,z); # y and z defined at all positions on the grid
        x=x[config.x_slice]*np.ones(np.shape(y));    # x is defined to be x at x_slice       
   

    wind_speed = config.wind_speed * np.ones_like(times) #m/s
    if config.wind_type == WindType.CONSTANT:
        wind_dir = np.zeros_like(times)
        wind_label = "Constant wind"
    elif config.wind_type == WindType.FLUCTUATING:
        wind_dir = 360. * np.random.rand(len(times))
        wind_label = "Fluctuating wind"
    elif config.wind_type == WindType.PREVAILING:
        wind_dir = -np.sqrt(2.) * erfcinv(2. * np.random.rand(len(times))) * 40.
        wind_dir = np.mod(wind_dir, 360)
        wind_label = "Prevailing wind"
    else:
        raise ValueError("Unsupported wind type")
        
    C1=np.zeros((len(x),len(y),len(wind_dir)))
    for t in range(len(times)):
        for (x_s, y_s, q, h) in config.stacks:
            C = gauss_func(q, wind_speed[t], wind_dir[t], x_grid, y_grid, z_grid,
                        x_s, y_s, h, stability[t])
            C1[:, :, t] += C
    #C1 storia di come si disperde

    if config. humidify:
        C1= apply_hygroscopic_growth(C1, config.RH, config.dry_size, config.aerosol_type)

    return C1, (x_grid, y_grid, z_grid), times, stability, wind_dir, stability_label, wind_label

