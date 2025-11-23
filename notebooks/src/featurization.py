import pandas as pd
import numpy as np

def calcualte_cone_parameters(event_data: pd.DataFrame) -> pd.Series:
    # min plane is the lowest non-zero plane
    min_plane = event_data[event_data['plane'] > 0]['plane'].min()
    max_plane = 0
    
    # get average x,y,z from min plane
    X_mean_min = event_data[event_data['plane'] == min_plane]['X_transformed'].mean()
    Y_mean_min = event_data[event_data['plane'] == min_plane]['Y_transformed'].mean()
    Z_mean_min = event_data[event_data['plane'] == min_plane]['Z_transformed'].mean()

    # get average x, y, z from max plane
    X_mean_max = event_data[event_data['plane'] == max_plane]['X_transformed'].mean()
    Y_mean_max = event_data[event_data['plane'] == max_plane]['Y_transformed'].mean()
    Z_mean_max = event_data[event_data['plane'] == max_plane]['Z_transformed'].mean()

    # radius is 3* l2 norm of std (x,y,z) at max plane
    radius = 3*np.linalg.norm([
        event_data[event_data['plane'] == max_plane]['X_transformed'].std(),
        event_data[event_data['plane'] == max_plane]['Y_transformed'].std(),
        event_data[event_data['plane'] == max_plane]['Z_transformed'].std()
    ])
    
    return pd.Series({
        'min_plane': min_plane,
        'max_plane': max_plane,
        'X_mean_min': X_mean_min,
        'Y_mean_min': Y_mean_min,
        'Z_mean_min': Z_mean_min,
        'X_mean_max': X_mean_max,
        'Y_mean_max': Y_mean_max,
        'Z_mean_max': Z_mean_max,
        'radius': radius
    })