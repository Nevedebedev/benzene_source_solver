# Benzene Source Solver

## Overview
2D FEniCS advection-diffusion solver training data for benzene dispersion PINN (Physics-Informed Neural Network).

## Files
- `testrun.py`: Main simulation script for generating training data
- `advection_diffusion_solver/`: Core solver modules
- `data/simulations/`: Complete training data for all scenarios (1-100)

## Data Structure (Per Scenario)
```
data/simulations/scenario_XXXX/
├── collocation_points_final.npz    # Main training data (11 columns)
├── collocation_points_partial_*.npz # Partial saves (if any)
├── ic_points.npz                   # Initial condition points
└── scenario_XXXX_log.txt           # Simulation log
```

## Data Format
- **Collocation points**: `[t, x, y, source_x, source_y, diameter, Q_total, wind_u, wind_v, D, phi]`
- **IC points**: Same format, but `t=0` and `phi=0`

## Usage
```bash
# Generate new scenarios
python testrun.py --scenario-id 91

# Run all scenarios
python testrun.py

# Load training data
import numpy as np
data = np.load('data/simulations/scenario_0091/collocation_points_final.npz')
training_data = data['data']  # Shape: [N_samples, 11]
```

## Physics Model
2D advection-diffusion-reaction equation:
```
∂φ/∂t + v·∇φ - D∇²φ = S
```
where:
- φ: concentration (kg/m³)
- v: wind velocity field (m/s)
- D: diffusion coefficient (m²/s)
- S: source term (kg/s·m²)

## Scenario Parameters
- **Domain**: 20 km × 20 km
- **Sources**: 10 different facility sizes (150-4500m diameter)
- **Wind speeds**: 0.5-10 m/s
- **Stability classes**: A-F (atmospheric conditions)
- **Emission rates**: 0.02-15.0 kg/s

## Requirements
- FEniCS/DOLFINx
- NumPy, SciPy, Matplotlib
- Python 3.8+

## Complete Dataset
This repository contains the complete dataset with all 100 scenarios covering:
- 10 different source sizes (150-4500m diameter)
- 10 different wind conditions (0.5-10 m/s)
- Various atmospheric stability classes
- Complete IC and collocation points for PINN training

Each scenario provides thousands of training points with full physics parameters.
