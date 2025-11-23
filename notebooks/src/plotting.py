import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
import os
import imageio

def plot_2d_scatter(train_data):
    """
    Plot a 2D scatter plot of X_transformed and Y_transformed from the training data.
    
    Parameters:
    train_data (pd.DataFrame): DataFrame containing 'X_transformed' and 'Y_transformed' columns for several events.
    
    Returns:
    figure, ax: Matplotlib figure and axis objects.
    """
    # Create a 2D scatter plot
    figure, ax = plt.subplots(figsize=(15, 10))
    plt.scatter(train_data['X_transformed'].values, train_data['Y_transformed'].values, s=0.01)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    # plt.xlim(-3, 3)
    # plt.ylim(-2, 2)
    plt.title('2D plot of X, Y with Z as color')
    plt.show()
    
    return figure, ax


def plot_histogram(train_data, bins=50):
    """
    Plot a histogram of a specified column from the data.
    
    Parameters:
    train_data (pd.DataFrame): DataFrame containing the column to plot.
    bins (int): Number of bins for the histogram.
    
    Returns:
    np.ndarray(Matplotlib.axes.Axes): The axes of the histogram plot.
    
    """
    axes = train_data.hist(bins=bins, figsize=(20,15))
    plt.tight_layout()
    plt.show()
    
    return axes


def plot_records_per_plane(train_data):
    """
    Plot a histogram showing the number of records per plane.
    
    Parameters:
    train_data (pd.DataFrame): DataFrame containing a 'plane' column.
    
    Returns:
    None
    """
    # group input data from all events in the data by 'plane' 
    # and count the number of records
    records_per_plane = train_data.groupby('plane').size().reset_index(name='num_records')

    # plot historgram of records per plane
    plt.figure(figsize=(10,6))
    plt.bar(records_per_plane['plane'], records_per_plane['num_records'])
    plt.xlabel('Plane')
    plt.ylabel('Number of Records')
    plt.title('Number of Records per Plane')
    plt.show()
    

def plot_cone(bottom_radius, tip_position, bottom_center, num_points=100):
    """
    Plot a 3D cone with customizable tip and bottom center positions.
    
    Parameters:
    -----------
    bottom_radius : float
        Radius of the cone's base
    tip_position : tuple or array
        (x_tip, y_tip, z_tip) coordinates of the cone's apex
    bottom_center : tuple or array
        (x_base, y_base, z_base) coordinates of the cone's base center
    num_points : int
        Resolution of the mesh (default: 100)
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to numpy arrays
    tip = np.array(tip_position)
    base = np.array(bottom_center)
    
    # Calculate the axis vector (from tip to base center)
    axis_vector = base - tip
    height = np.linalg.norm(axis_vector)
    axis_unit = axis_vector / height
    
    # Create parametric grid
    theta = np.linspace(0, 2 * np.pi, num_points)
    h = np.linspace(0, height, num_points)
    Theta, H = np.meshgrid(theta, h)
    
    # Create perpendicular vectors for the circular cross-section
    # Find a vector perpendicular to the axis
    if abs(axis_unit[2]) < 0.9:
        perp1 = np.cross(axis_unit, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(axis_unit, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    
    # Get second perpendicular vector
    perp2 = np.cross(axis_unit, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    # Radius varies linearly from 0 at tip to bottom_radius at base
    R = (bottom_radius / height) * H
    
    # Generate points along the cone
    X = tip[0] + H * axis_unit[0] + R * (np.cos(Theta) * perp1[0] + np.sin(Theta) * perp2[0])
    Y = tip[1] + H * axis_unit[1] + R * (np.cos(Theta) * perp1[1] + np.sin(Theta) * perp2[1])
    Z = tip[2] + H * axis_unit[2] + R * (np.cos(Theta) * perp1[2] + np.sin(Theta) * perp2[2])
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, color='red', alpha=0.2, 
                           edgecolor='none', antialiased=True)
    
    # Optional: Add base circle (cap)
    theta_cap = np.linspace(0, 2 * np.pi, num_points)
    x_cap = base[0] + bottom_radius * (np.cos(theta_cap) * perp1[0] + np.sin(theta_cap) * perp2[0])
    y_cap = base[1] + bottom_radius * (np.cos(theta_cap) * perp1[1] + np.sin(theta_cap) * perp2[1])
    z_cap = base[2] + bottom_radius * (np.cos(theta_cap) * perp1[2] + np.sin(theta_cap) * perp2[2])
    
    # Create triangulation for the base
    r_cap = np.linspace(0, bottom_radius, 20)
    theta_cap_mesh, r_cap_mesh = np.meshgrid(theta_cap, r_cap)
    x_cap_mesh = base[0] + r_cap_mesh * (np.cos(theta_cap_mesh) * perp1[0] + np.sin(theta_cap_mesh) * perp2[0])
    y_cap_mesh = base[1] + r_cap_mesh * (np.cos(theta_cap_mesh) * perp1[1] + np.sin(theta_cap_mesh) * perp2[1])
    z_cap_mesh = base[2] + r_cap_mesh * (np.cos(theta_cap_mesh) * perp1[2] + np.sin(theta_cap_mesh) * perp2[2])
    ax.plot_surface(x_cap_mesh, y_cap_mesh, z_cap_mesh, color='blue', alpha=0.5)
    
    # Add labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(f'3D Cone\nRadius: {bottom_radius:.2f}, Height: {height:.2f}\n' + 
                 f'Tip: {tuple(np.round(tip, 2))}, Base: {tuple(np.round(base, 2))}')
      
    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # set limits
    ax.set_xlim([50, 350])
    ax.set_ylim([-100, 200])
    # ax.set_zlim([-2, 1.5])
    
    return fig, ax


def plot_cone_with_scatter(top_event_data, elev=10., azim=330.0):
    
    min_plane = top_event_data[top_event_data['plane'] > 0]['plane'].min()
    max_plane = top_event_data['plane'].max()
    
    # plot 3d cone here
    # get average x,y,z from plane 0
    avg_x_min = top_event_data[top_event_data['plane'] == min_plane]['X_transformed'].mean()
    avg_y_min = top_event_data[top_event_data['plane'] == min_plane]['Y_transformed'].mean()
    avg_z_min = top_event_data[top_event_data['plane'] == min_plane]['Z_transformed'].mean()

    # get average x, y, z from plane 24
    avg_x_max = top_event_data[top_event_data['plane'] == max_plane]['X_transformed'].mean()
    avg_y_max = top_event_data[top_event_data['plane'] == max_plane]['Y_transformed'].mean()
    avg_z_max = top_event_data[top_event_data['plane'] == max_plane]['Z_transformed'].mean()

    # radius is 3* l2 norm of std (x,y,z) at plane 24
    radius = 3*np.linalg.norm([
        top_event_data[top_event_data['plane'] == max_plane]['X_transformed'].std(),
        top_event_data[top_event_data['plane'] == max_plane]['Y_transformed'].std(),
        top_event_data[top_event_data['plane'] == max_plane]['Z_transformed'].std()
    ])

    fig, ax = plot_cone(
        bottom_radius=radius, 
        tip_position=(avg_x_min, avg_y_min, avg_z_min), 
        bottom_center=(avg_x_max, avg_y_max, avg_z_max)
    )
    
    # add scatter plot of the hits
    tmin = top_event_data['time_transformed'].min()
    tmax = top_event_data['time_transformed'].max()
    cmap = plt.get_cmap('viridis')

    # Scatter plot
    # plot 3d particle plot where x, y, z are X_transformed, Y_transformed, Z_transformed and color depends on kinetic energy
    sc = ax.scatter(
            top_event_data['X_transformed'], 
            top_event_data['Y_transformed'], 
            top_event_data['Z_transformed'],
            color=cmap((top_event_data["time_transformed"] - tmin) / (tmax - tmin)),
        )

    cbar = plt.colorbar(sc, label='Normalized Time', pad = 0.1, shrink=0.5)
    ax.set_xlabel('X_transformed')
    ax.set_ylabel('Y_transformed')
    ax.set_zlabel('Z_transformed')
    ax.view_init(elev=elev, azim=azim)

    return fig, ax

def plot_y_values_per_plane(top_event_data):
    """
    Plot Y_transformed values for each plane in a single plot.
    
    Parameters:
    top_event_data (pd.DataFrame): DataFrame containing 'plane' and 'Y_transformed' columns for a specific event.
    
    Returns:
    figure: Matplotlib figure object.
    """
    # validate inputs
    if 'plane' not in top_event_data.columns or 'Y_transformed' not in top_event_data.columns:
        raise ValueError("Input DataFrame must contain 'plane' and 'Y_transformed' columns.")
    
    # plot y values plot foe each aplne in a single plot
    figure = plt.figure(figsize=(10,6))
    for plane in sorted(top_event_data['plane'].unique()):
        plane_data = top_event_data[top_event_data['plane'] == plane].copy()
        # smoothen
        plane_data['Y_transformed'] = plane_data['Y_transformed'].rolling(window=50).mean()
        plt.plot(plane_data['Y_transformed'], label=f'Plane {plane}')
    plt.xlabel('Index')
    plt.ylabel('Y Transformed')
    plt.title('Y Transformed for Each Plane')
    plt.legend()
    plt.show()  
    return figure

def generate_3d_cone_animation(event_data, output_folder="cone_frames"):
    """
    Fit a cone into a particle distribution and plot animation for different azimuth angles for the given event data.
    
    Parameters:
    event_data (pd.DataFrame): DataFrame containing 'X_transformed', 'Y_transformed', 'Z_transformed', and 'plane' columns for a specific event.
    output_folder (str): Folder to save the animation frames.
    
    Returns:
    None
    """
    # validate inputs
    required_columns = ['X_transformed', 'Y_transformed', 'Z_transformed', 'plane']
    for col in required_columns:
        if col not in event_data.columns:
            raise ValueError(f"Input DataFrame must contain '{col}' column.")
        
    # save each frame as png and create animation later in a new folder
    os.makedirs(output_folder, exist_ok=True)

    for angle in range(0, 360, 5):
        fig, ax = plot_cone_with_scatter(event_data, elev=15., azim=angle)
        plt.savefig(os.path.join(output_folder, f'cone_frame_{angle:03d}.png'))
        plt.close(fig)
        
    # create animation from the png files using imageio
    images = []
    for angle in range(0, 360, 5):
        filename = os.path.join(output_folder, f'cone_frame_{angle:03d}.png')
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(output_folder,'cone_animation.gif'), images, loop=0, fps=60)
