import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import minimize
import warnings
from typing import Any
from tqdm.notebook import tqdm
from matplotlib import animation
warnings.filterwarnings('ignore')
 
class Simulator():  # Fixed typo from "Simualtor"
    """
        inputs:
            takes the output of the diffusion pipeline which is a shower (x, y, N, T)
            number_of_detectors = 64
            detector_radius = 5m
            mountain_size = (20,20)
        
        detectors:
            model detectors as circles in the mountain (represented as a 2d space of a rectangle plane),
            model is a circle in the rectangular mesh, which is a mountain
            
        shower power:
            shower will hit the plane and evaluate how much will be received by each detector
       
        detector power:
            then we assign total energy for each detector
            sum of energies of the shower particles,
        
        optimization:
            a target metric is needed to optimize against this; use total energy for now 
        
        output:
            depending on this assign xy coordinates in the plane for recommended detectors
        
            figures:
                plot that the sum of the energy of all detectors
                = <the energy of the shower> - <energy outside the detectors>
    """
    
    def __init__(self, number_of_detectors=64, detector_radius=5, mountain_x=(200,800), mountain_y=(-200,300), csv_path=None):
        self.number_of_detectors = number_of_detectors
        self.detector_radius = detector_radius
        self.mountain_x = mountain_x
        self.mountain_y = mountain_y
        self.csv_path = csv_path
        self.detectors_loc = []
        self.inputs = None
        self.detected_particles = []
        self.total_particles = []
        
    def read_data(self, csv_path, drop_outliers=False):
        """Load data from CSV file"""
        if self.inputs is None:    
            self.csv_path = csv_path
            print(f"Loading data from {self.csv_path}...")
            
            # Load from .csv
            try:
                inputs_data = pd.read_csv(self.csv_path)
                # save names for inputs and labels
                input_names = ["X_transformed", "Y_transformed", "kinetic_energy", "time"]
                
                if drop_outliers:
                    # Remove outliers based on kinetic_energy
                    q_low = inputs_data["kinetic_energy"].quantile(0.01)
                    q_high = inputs_data["kinetic_energy"].quantile(0.99)
                    inputs_data = inputs_data[(inputs_data["kinetic_energy"] >= q_low) & (inputs_data["kinetic_energy"] <= q_high)]
                    print(f"Outliers removed: keeping data between {q_low} and {q_high} kinetic energy.")

                # convert to pandas dataframe
                self.inputs = inputs_data[input_names]
                
                print("Data has been successfully loaded")
                print(f"Shape: {self.inputs.shape}")
            
            except Exception as e:
                print(f"Error during data loading: {e}")
        else:
            print("Data is already loaded")
            
    def initialize_detectors(self, initialization_logic="rand"):
        """Initialize detector locations"""
        if initialization_logic == "rand":
            # Generate all random coordinates at once
            rng = np.random.default_rng(seed=1234)

            x_coords = rng.uniform(self.mountain_x[0], self.mountain_x[1], self.number_of_detectors)
            y_coords = rng.uniform(self.mountain_y[0], self.mountain_y[1], self.number_of_detectors)
            
            # Stack into a 2D array
            self.detectors_loc = np.column_stack((x_coords, y_coords))        
            print("Random initialization completed.")
    
        elif initialization_logic == "uniform":  # Fixed from "else if"
            # Create a grid of detectors
            n_side = int(np.sqrt(self.number_of_detectors))
            if n_side * n_side < self.number_of_detectors:
                n_side += 1

            x_step = (self.mountain_x[1] - self.mountain_x[0]) / n_side
            y_step = (self.mountain_y[1] - self.mountain_y[0]) / n_side

            # Pre-allocate array with exact size needed
            self.detectors_loc = np.zeros((self.number_of_detectors, 2))

            idx = 0
            for i in range(n_side):
                for j in range(n_side):
                    if idx < self.number_of_detectors:
                        x = (i + 0.5) * x_step + self.mountain_x[0]
                        y = (j + 0.5) * y_step + self.mountain_y[0]
                        self.detectors_loc[idx] = [x, y]
                        idx += 1
                    else:
                        break
                if idx >= self.number_of_detectors:
                    break
            
            print("Uniform initialization completed.")
            
        else: 
            print("Initialization logic string not recognized.")
              
    def calculate_detected_particles(self):
        """
        For each row in the input array, calculate if the particles have been detected 
        by comparing their location to the detector centers, save data for each timestep.
        Vectorized for performance.
        """
        if self.inputs is None:
            print("No input data loaded. Please run read_data() first.")
            return

        self.detected_particles = []

        # Convert detectors to numpy array for vectorized distance calculation
        detectors = self.detectors_loc
        det_x = detectors[:, 0]
        det_y = detectors[:, 1]
        det_radius_sq = self.detector_radius ** 2

        # Group by event/time
        for event_id, event_data in self.inputs.groupby('time'):
            # Extract particle positions and energies
            x = event_data['X_transformed'].values
            y = event_data['Y_transformed'].values
            energy = event_data['kinetic_energy'].values

            # Filter particles within mountain bounds (vectorized)
            in_bounds = (
                (x >= self.mountain_x[0]) & (x <= self.mountain_x[1]) &
                (y >= self.mountain_y[0]) & (y <= self.mountain_y[1])
            )
            x = x[in_bounds]
            y = y[in_bounds]
            energy = energy[in_bounds]

            if len(x) == 0:
                self.detected_particles.append(0)
                continue

            # Compute squared distances to all detectors for all particles
            # Shape: (num_particles, num_detectors)
            dx = x[:, None] - det_x[None, :]
            dy = y[:, None] - det_y[None, :]
            dist_sq = dx**2 + dy**2

            # For each particle, check if it is within any detector's radius
            detected_mask = (dist_sq <= det_radius_sq).any(axis=1)

            detected_energy = energy[detected_mask].sum()
            self.detected_particles.append(detected_energy)

        # print(f"Calculated detected particles for {len(self.detected_particles)} events")
         
    def calculate_total_particles(self):
        """
        Sum all particles in the input array for each timestep (vectorized with numpy)
        """
        if self.inputs is None:
            print("No input data loaded. Please run read_data() first.")
            return

        # Use groupby and sum for vectorized computation
        grouped = self.inputs.groupby('time')['kinetic_energy'].sum()
        self.total_particles = grouped.values.tolist()
        
        print(f"Calculated total particles for {len(self.total_particles)} events")
        
    def plot_energies(self):
        """
        Plot:
            for each timestep the recorded energy vs missed particles
            3d plot of the detectors and the particle hits
        Print:
            total energy
            detected energy
        """
        if not self.detected_particles or not self.total_particles:
            print("Please run calculate_detected_particles() and calculate_total_particles() first.")
            return
        
        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 12))
        
        # Plot 1: Energy comparison
        events = range(len(self.total_particles))
        # missed_energy = [total - detected for total, detected in zip(self.total_particles, self.detected_particles)]
        
        ax1.semilogy(events, self.total_particles, label='Total Energy', alpha=0.7)
        ax1.semilogy(events, self.detected_particles, label='Detected Energy', alpha=0.7)
        # ax1.semilogy(events, missed_energy, label='Missed Energy', alpha=0.7)
        ax1.set_xlabel('Event')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy Detection per Event')
        ax1.legend()
        ax1.grid(True)
        
    def plot_detector_layout(self):
        """
        Plot:
            2d plot of the detectors and the particle hits
        Print:
            total energy
            detected energy
        """
        fig, (ax2) = plt.subplots(1, 1, figsize=(15, 12))

        # Plot 2: Detector layout and sample particle hits
        ax2.set_xlim(self.mountain_x[0], self.mountain_x[1])
        ax2.set_ylim(self.mountain_y[0], self.mountain_y[1])

        # Plot detectors as circles
        for det_x, det_y in self.detectors_loc:
            circle = Circle((det_x, det_y), self.detector_radius, alpha=0.3, color='blue')
            ax2.add_patch(circle)
        
        # Plot sample particle hits from first event
        if self.inputs is not None:
            # sample_event = self.inputs.sample(frac=0.01)  # Sample 1% for visibility
            sample_event = self.inputs
            ax2.scatter(sample_event['X_transformed'], sample_event['Y_transformed'], 
                       s=sample_event['kinetic_energy']/100, c='red', label='Particles')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Detector Layout and Particle Hits (Event 0)')
        ax2.grid(True)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
    def print_summary_statistics(self):
        """ 
        Print summary statistics
        """
        total_energy = sum(self.total_particles)
        detected_energy = sum(self.detected_particles)
        detection_efficiency = detected_energy / total_energy * 100 if total_energy > 0 else 0
        
        print(f"\n=== Energy Summary ===")
        print(f"Total Energy: {total_energy:.2f}")
        print(f"Detected Energy: {detected_energy:.2f}")
        print(f"Detection Efficiency: {detection_efficiency:.2f}%")
        
        
    def optimize_setup_sgd(self, learning_rate=1, batch_size=1000, n_epochs=100, momentum=0.9):
        """
        Optimize detector positions using Stochastic Gradient Descent
        
        Parameters:
        - learning_rate: Step size for gradient updates
        - batch_size: Number of events to use per gradient calculation
        - n_epochs: Number of complete passes through the data
        - momentum: Momentum factor to accelerate convergence
        """
        print(f"Starting SGD optimization with learning rate {learning_rate}, batch_size {batch_size}")
        
        if self.inputs is None:
            print("No input data loaded. Please run read_data() first.")
            return
        
        detectors = self.detectors_loc.copy()
        
        # Initialize momentum terms
        print("Initializing momentum terms...")
        velocity = np.zeros_like(detectors)
        
        # Group events by time for batch processing
        print("Grouping events for batch processing...")
        events = list(self.inputs.groupby('time'))
        n_events = len(events)
        
        best_efficiency = 0
        best_detectors = detectors.copy()
        efficiency_history = []
        
        # For animation
        location_history = np.array([detectors.copy()])
            
        for epoch in tqdm(range(n_epochs), desc="SGD Epochs"):
            epoch_loss = 0
            n_batches = 0
            
            # Process mini-batches
            for batch_start in tqdm(range(0, n_events, batch_size), desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
                batch_end = min(batch_start + batch_size, n_events)
                batch_events = events[batch_start:batch_end]
                
                # Calculate gradient for this batch
                gradient, batch_loss = self._calculate_gradient_batch(detectors, batch_events)

                print(np.mean(gradient))

                # Update velocities with momentum
                # velocity = momentum * velocity - learning_rate * gradient
                velocity = 10 * learning_rate * gradient/batch_size
                
                # Update detector positions
                detectors += velocity
                
                # Apply boundary constraints
                detectors[:, 0] = np.clip(detectors[:, 0], self.mountain_x[0], self.mountain_x[1])
                detectors[:, 1] = np.clip(detectors[:, 1], self.mountain_y[0], self.mountain_y[1])
                
                epoch_loss += batch_loss
                n_batches += 1
                location_history = np.vstack((location_history, [detectors.copy()]))
            
            # Calculate current efficiency for monitoring
            self.detectors_loc = detectors.copy()
            self.calculate_detected_particles()
            
            if self.total_particles:
                current_efficiency = sum(self.detected_particles) / sum(self.total_particles)
                efficiency_history.append(current_efficiency)
                
                if current_efficiency > best_efficiency:
                    best_efficiency = current_efficiency
                    best_detectors = detectors.copy()
            
            # Print progress
            avg_loss = epoch_loss / n_batches
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}, "
                f"Efficiency: {current_efficiency:.4f}")
        
        print(f"SGD optimization completed. Best efficiency: {best_efficiency:.4f}")


        return efficiency_history, location_history

    def _calculate_gradient_batch(self, detectors, batch_events):
        """
        Parameters:
        - detectors: Current detector positions (numpy array of shape [num_detectors, 2])
        - batch_events: List of (event_id, event_data) tuples for the current batch
        Returns:
        - total_gradient: Gradient of the loss w.r.t. detector positions
        - total_loss: Total loss for the batch
        
        Calculate gradient of the loss function with respect to detector positions
        
        Loss function: negative detection efficiency (we want to maximize detection)
        """
        total_gradient = np.zeros_like(detectors)
        total_loss = 0
        
        for event_id, event_data in batch_events:
            # Extract particle data
            x = event_data['X_transformed'].values
            y = event_data['Y_transformed'].values
            energy = event_data['kinetic_energy'].values
            
            if len(x) == 0:
                continue
            
            # Calculate gradient for each detector
            for det_idx in range(len(detectors)):
                det_x, det_y = detectors[det_idx]
                
                # Distance from particles to detector
                dx = x - det_x
                dy = y - det_y
                dist_sq = dx**2 + dy**2
                dist = np.sqrt(dist_sq)
                
                # Gradient of soft detection with respect to detector position
                grad_x, grad_y = self._detection_gradient(dx, dy, dist, energy, self.detector_radius)
                
                total_gradient[det_idx, 0] += grad_x
                total_gradient[det_idx, 1] += grad_y
            
            # Loss: negative total detected energy (we want to maximize detection)
            total_detected = sum(energy[
                ((x[:, None] - detectors[:, 0])**2 + (y[:, None] - detectors[:, 1])**2 
                <= self.detector_radius**2).any(axis=1)
            ])
            total_loss -= total_detected / len(batch_events)
        
        return total_gradient, total_loss

    def _detection_gradient(self, dx, dy, dist, energy, radius):
        """
        Parameters:
        - dx, dy: Distances from particles to detector center
        - dist: Euclidean distance from particles to detector center
        - energy: Particle energies
        - radius: Detector radius
        Returns:
        - grad_x, grad_y: Gradients of detection efficiency w.r.t. detector position
        """

        # Avoid division by zero
        dist = np.maximum(dist, 1e-8)
        
        # Gradient of soft detection function
        steepness = 1
        sigmoid_val = 1 / (1 + np.exp(-steepness * (dist - radius)))
        sigmoid_grad = steepness * sigmoid_val
        
        # Chain rule: gradient w.r.t. detector position
        grad_x = np.sum(energy * sigmoid_grad * (dx / dist))
        grad_y = np.sum(energy * sigmoid_grad * (dy / dist))

        return grad_x, grad_y
