import torch
import torch.optim as optim
import numpy as np
from torchdiffeq import odeint
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, ITRS, GCRS
from astropy.time import Time
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


#################################################################################
# Constants
# 1. Setup Constants using Astropy
MU = const.GM_earth.to(u.m**3 / u.s**2).value  # GM
R_E = const.R_earth.to(u.m).value              # Earth Equatorial Radius

# Zonal Harmonics (WGS84 / EGM96 typical values)
J_COEFFS = {
    2: 1.0826263e-3,
    3: -2.532656e-6,
    4: -1.619620e-6,
    5: -0.227296e-6,
    6: 0.540681e-6
}

#################################################################################
def thrust_model(time):
    # Placeholder for thrust model
    # X- is sinusoidal thrust in +x direction with amplitude 2e-5 m/s² and period of 24 hours
    x_thrust = 1.80e-5 * torch.sin(2 * np.pi * time / (24 * 3600))
    # Y- is sinusoidal thrust in +y direction with amplitude 2e-5 m/s² and period of 24 hours shifted by 12 hours
    y_thrust = 2.50e-5 * torch.sin(2 * np.pi * (time - 14*3600) / (24 * 3600)) 
    #Z- is sinusoidal thrust in +z direction with amplitude 4e-5 m/s² and period of 24 hours shifted by 8 hours
    z_thrust = 4.85e-5 * torch.sin(2 * np.pi * (time + 8*3600) / (24 * 3600))

    return torch.tensor([x_thrust, y_thrust, z_thrust], dtype=torch.float64)  # m/s²

#################################################################################
# Define the physics model for use in ODE solver
def physics_model(time, state):
    # state X = [rx, ry, rz, vx, vy, vz]
    r_vec = state[0:3]
    v_vec = state[3:6]
    
    r_mag = torch.norm(r_vec)
    z = r_vec[2]
            
    # --- Keplerian Acceleration (Two-Body) ---
    a_kepler = -MU * r_vec / (r_mag**3)
    
    # --- Perturbing Acceleration (J2) ---
    
    # Explicit Cartesian components for J2 (High performance)
    # Based on Equation 4 context
    j2_factor = (3/2) * J_COEFFS[2] * MU * (R_E**2) / (r_mag**4)
    z2_r2 = (z**2) / (r_mag**2)
    a_j2 = j2_factor * torch.stack([
        (r_vec[0]/r_mag) * (5*z2_r2 - 1),
        (r_vec[1]/r_mag) * (5*z2_r2 - 1),
        (r_vec[2]/r_mag) * (5*z2_r2 - 3)
    ])

    # Here we add J2 
    total_accel = a_kepler + a_j2 

    return torch.cat([v_vec, total_accel])

###############################################################################
# ODE Model for Satellite Dynamics assuming no thrust and only J2 perturbation
class SatelliteODE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, state):
        # state X = [rx, ry, rz, vx, vy, vz]
        r_vec = state[0:3]
        v_vec = state[3:6]
        
        a_grav = physics_model(t, state)[3:6]
        
        return torch.cat([v_vec, a_grav]) 

#############################################################################

# ODE assuming continuous thrust to generate the data
class SatelliteODEThrust(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, state):
        # state X = [rx, ry, rz, vx, vy, vz]
        r_vec = state[0:3]
        v_vec = state[3:6]
        
        # Get gravitational acceleration including J2
        a_grav = physics_model(t, state)[3:6]

        # Generate thrust acceleration (example: constant thrust in +x direction)
        a_thrust = thrust_model(t)

        total_accel = a_grav + a_thrust
        
        return torch.cat([v_vec, total_accel])
    
# ---------------------------------------------------------------------------
# Define a "physics only" solver
class SatelliteODEPhysicsOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, state):

        with torch.set_grad_enabled(True):
            # state X = [rx, ry, rz, vx, vy, vz]
            r_vec = state[0:3]
            v_vec = state[3:6]
            
            a_grav = physics_model(t, state)[3:6]
            
            return torch.cat([v_vec, a_grav])
        
# ---------------------------------------------------------------------------
# Define class for creating MLP with inputs (number of layers, neurons per layer, activation function)
class MLP(torch.nn.Module):
    def __init__(self, input_size = 7, output_size = 3, hidden_layers = 2, neurons_per_layer = 100):
        super(MLP, self).__init__()

        R_REF = 42164140.0
        V_REF = 3074.6
        scaling = torch.tensor([R_REF, R_REF, R_REF, V_REF, V_REF, V_REF], dtype=torch.float64)

        # Enforce float64 to match your ODE solver
        self.input_size = input_size
        self.scaling = scaling
        
        layers = []
        layers.append(torch.nn.Linear(input_size, neurons_per_layer))
        layers.append(torch.nn.Tanh()) # Tanh is preferred for PINNs (smooth second derivatives)

        for _ in range(hidden_layers - 1):
            layers.append(torch.nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(torch.nn.Tanh())

        layers.append(torch.nn.Linear(neurons_per_layer, output_size))
        self.network = torch.nn.Sequential(*layers).to(torch.float64)

        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self, t, state):
        # Concatenate time and state into a single input vector [t, x, y, z, vx, vy, vz]
        # t is often a scalar, so we expand it
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # FEATURE SCALING: Crucial for PINNs
        # Normalize t by mission duration and state by GEO radius/velocity
        # This keeps inputs around [-1, 1]
        t_norm = t / 172800.0 # Example: 48 hours in seconds
        state_norm = state / self.scaling # Using previously defined scaling tensor
        
        x_input = torch.cat([t_norm, state_norm])
        
        # The output is the 'Anomalous Acceleration' (g)
        return self.network(x_input) * 1e-7  # Scale output to realistic acceleration magnitudes


# Define a hybrid ODE model combining physics and neural network
class PINN(torch.nn.Module):
    def __init__(self, neural_net):
        super().__init__()
        self.neural_net = neural_net

    def forward(self, t, state):
        # state X = [rx, ry, rz, vx, vy, vz]
        r_vec = state[0:3]
        v_vec = state[3:6]
        
        # Get gravitational acceleration including J2
        a_grav = physics_model(t, state)[3:6]
        
        # Get neural network acceleration prediction
        a_nn = self.neural_net(t, state)
        
        # Combine physics and neural network
        total_accel = a_grav + a_nn
        
        return torch.cat([v_vec, total_accel])