"""
2D FEniCS Data Generation for Benzene Dispersion
Ground-level dispersion modeling with source diameter parameter
Samples at 20s intervals for PINN training
"""

import numpy as np
from pathlib import Path
import json
from datetime import datetime
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl
from dataclasses import dataclass
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
import sys
import argparse
import fcntl
import logging
import signal
import sys

from advection_diffusion_solver.atmospheric_physics import (
    sample_meteorological_conditions,
    get_pasquill_class,
    stability_class_to_D,
    check_steady_state_mass
)


# Configuration

@dataclass
class SimulationParams:
    Lx: float = 20000.0  # Changed from 40000.0 to 20000.0 (20km)
    Ly: float = 20000.0  # Changed from 40000.0 to 20000.0 (20km)
    nx: int = 200  # Changed from 250 to 200 for 20km domain
    ny: int = 200  # Changed from 250 to 200 for 20km domain
    T: float = 500.0
    dt: float = 0.1           # Time step (s) - may be adjusted per scenario
    output_freq: int = 200     # Save every 200 timesteps (20s with dt=0.1)
    z0: float = 0.03          # Surface roughness (m)
    
    # Source diameter range and emission rate dependencies
    diameter_min: float = 150.0   # 150m - small area source
    diameter_max: float = 4500.0  # 4500m - very large distributed source
    
    # Emission rate categories based on diameter
    # Small (150-1500m): 0.02-0.78 kg/s (avg 0.34 kg/s)
    # Medium (1500-2500m): 1.9-4.3 kg/s (avg 2.9 kg/s)
    # Large (2500-4500m): 7.0-15.0 kg/s (avg 10.0 kg/s)
    Q_total_min: float = 0.02    # Smallest facility (kg/s)
    Q_total_max: float = 15.0    # Largest facility (kg/s)
    Q_log_scale: bool = False    # Use uniform within categories
    
    supg_factor: float = 0.5
    sampling_margin: float = 200.0  # Reduced for 20km domain

    # Time control: run until plume reaches domain boundary
    T_min: float = 100.0     # Minimum runtime before checking boundary (reduced)
    T_max: float = 10000.0   # Maximum runtime (increased for boundary detection)
    boundary_threshold: float = 1e-7  # Concentration threshold for boundary detection (kg/m³)
    boundary_check_interval: int = 50  # Check boundary more frequently

@dataclass
class WindCondition:
    speed: float
    direction: float
    
    @property
    def u(self): 
        return self.speed * np.cos(np.radians(self.direction))
    
    @property
    def v(self): 
        return self.speed * np.sin(np.radians(self.direction))

@dataclass
class SourceLocation:
    id: int
    x: float
    y: float
    name: str = ""

# Scenario generation

def generate_source_locations(params: SimulationParams) -> List[SourceLocation]:
    """10 strategic source positions for 20km×20km domain"""
    Lx, Ly = params.Lx, params.Ly
    cx, cy = Lx / 2.0, Ly / 2.0
    ox, oy = 0.2 * Lx, 0.2 * Ly  # Increased offset for smaller domain
    return [
        SourceLocation(1, cx - ox, cy - oy, "SW_quad"),
        SourceLocation(2, cx - ox, cy, "W_mid"),
        SourceLocation(3, cx - ox, cy + oy, "NW_quad"),
        SourceLocation(4, cx, cy - oy, "S_mid"),
        SourceLocation(5, cx, cy, "Center"),
        SourceLocation(6, cx, cy + oy, "N_mid"),
        SourceLocation(7, cx + ox, cy - oy, "SE_quad"),
        SourceLocation(8, cx + ox, cy, "E_mid"),
        SourceLocation(9, cx + ox, cy + oy, "NE_quad"),
        SourceLocation(10, cx, cy + 0.6 * oy, "N_inner"),  # Adjusted for smaller domain
    ]

def get_emission_rate_for_diameter(diameter, rng):
    """Get emission rate based on diameter category"""
    if diameter >= 2500:  # Large facilities
        Q_min, Q_max = 7.0, 15.0
    elif diameter >= 1500:  # Medium facilities
        Q_min, Q_max = 1.9, 4.3
    else:  # Small facilities
        Q_min, Q_max = 0.02, 0.78
    
    return rng.uniform(Q_min, Q_max)

def generate_scenario_manifest(params: SimulationParams) -> List[Dict]:
    """400 scenarios with diameter-dependent emission rates"""
    scenarios = []
    scenario_id = 0
    
    sources = generate_source_locations(params)
    
    # 10 wind conditions
    winds = [
        WindCondition(0.5, 0),
        WindCondition(0.5, 180),
        WindCondition(2.0, 45),
        WindCondition(2.0, 225),
        WindCondition(4.0, 90),
        WindCondition(4.0, 270),
        WindCondition(6.0, 135),
        WindCondition(6.0, 315),
        WindCondition(10.0, 0),
        WindCondition(10.0, 180),
    ]
    
    rng = np.random.default_rng(seed=42)
    
    # Category probabilities (from your statistics)
    category_probs = [0.593, 0.222, 0.185]  # small, medium, large
    category_ranges = [(150, 1500), (1500, 2500), (2500, 4500)]
    
    for source in sources:
        for wind in winds:
            scenario_id += 1
            
            # Sample diameter from category-based distribution
            category_idx = rng.choice(3, p=category_probs)
            d_min, d_max = category_ranges[category_idx]
            diameter = rng.uniform(d_min, d_max)
            
            # Get emission rate based on diameter category
            Q_total = get_emission_rate_for_diameter(diameter, rng)
            
            # Sample meteorology and calculate stability class
            solar, cloud, hour, is_day = sample_meteorological_conditions(
                wind.speed, rng
            )
            
            stability_class = get_pasquill_class(
                wind.speed, solar, cloud, is_day
            )
            
            # Calculate D from stability class (this is derived, not input)
            D_horizontal, D_vertical = stability_class_to_D(
                stability_class, wind.speed, params.z0
            )
            
            scenarios.append({
                'id': scenario_id,
                'geometry': 'open_field',
                'source_id': source.id,
                'source_x': source.x,
                'source_y': source.y,
                'source_name': source.name,
                'source_diameter': float(diameter),
                'Q_total': float(Q_total),
                'wind_speed': wind.speed,
                'wind_direction': wind.direction,
                'wind_u': wind.u,
                'wind_v': wind.v,
                'stability_class': stability_class,  # This is the input
                'D': float(D_horizontal),  # This is derived
                'solar_radiation': float(solar),
                'cloud_cover': float(cloud),
                'hour': float(hour),
            })
    
    # Print category statistics
    small_scenarios = sum(1 for s in scenarios if s['source_diameter'] < 1500)
    medium_scenarios = sum(1 for s in scenarios if 1500 <= s['source_diameter'] < 2500)
    large_scenarios = sum(1 for s in scenarios if s['source_diameter'] >= 2500)
    
    print(f"\nGenerated {len(scenarios)} scenarios (10 sources × 10 winds)")
    print(f"  Small facilities (150-1500m): {small_scenarios} scenarios")
    print(f"    Q range: 0.02-0.78 kg/s")
    print(f"  Medium facilities (1500-2500m): {medium_scenarios} scenarios")
    print(f"    Q range: 1.9-4.3 kg/s")
    print(f"  Large facilities (2500-4500m): {large_scenarios} scenarios")
    print(f"    Q range: 7.0-15.0 kg/s")
    print(f"  Stability classes: A-F (D derived from stability)")
    return scenarios

# Boundary conditions

def apply_boundary_conditions_2D(V, params):
    """
    Do-nothing BCs on all boundaries (natural inflow/outflow)
    Advection handles outflow automatically via weak form
    """
    return []

# Source

class GaussianSource2D:
    """2D Gaussian area source"""
    def __init__(self, x_source, y_source, Q_total, diameter):
        self.xs = x_source
        self.ys = y_source
        self.Q_total = Q_total
        self.diameter = diameter
        self.sigma = diameter / 4.0  # ~95% within diameter
        
        # Convert to area source intensity (kg/s/m²)
        self.Q0 = Q_total / (2 * np.pi * self.sigma**2)

    def __call__(self, x):
        r_squared = (x[0] - self.xs)**2 + (x[1] - self.ys)**2
        # Return in kg/s/m² - NO CONVERSION!
        return self.Q0 * np.exp(-r_squared / (2 * self.sigma**2))

# ADR Solver

class ADRSolver2D:
    """2D Advection-Diffusion solver"""
    
    def __init__(self, params, domain, scenario, source):
        self.params = params
        self.domain = domain
        self.scenario = scenario
        self.wind_u = scenario['wind_u']
        self.wind_v = scenario['wind_v']
        self.D = scenario['D']
        self.source = source
        
        # Function space
        self.V = fem.functionspace(domain, ("CG", 1))
        
        # Trial/test functions
        self.phi = ufl.TrialFunction(self.V)
        self.v_test = ufl.TestFunction(self.V)
        
        # Previous solution
        self.phi_n = fem.Function(self.V)
        self.phi_n.name = "phi"
        self.phi_n.x.array[:] = 0.0
        
        # Source function
        self.source_func = fem.Function(self.V)
        try:
            self.source_func.interpolate(self.source)
        except Exception as e:
            raise RuntimeError(f"Source interpolation failed: {str(e)}")
        
        # Boundary conditions
        self.bcs = apply_boundary_conditions_2D(self.V, params)
        
        # Setup problem
        self._setup_variational_problem()
    
    def _setup_variational_problem(self):
        """Setup 2D ADR weak form"""
        dt = self.params.dt
        D = self.D
        u = self.wind_u
        v = self.wind_v
        
        # Velocity
        vel = ufl.as_vector([u, v])
        vel_mag = ufl.sqrt(u**2 + v**2) + 1e-10
        
        # SUPG parameter
        h = ufl.CellDiameter(self.domain)
        tau = self.params.supg_factor * h / (2 * vel_mag)
        
        # Standard weak form
        F = (self.phi - self.phi_n)/dt * self.v_test * ufl.dx
        F += D * ufl.dot(ufl.grad(self.phi), ufl.grad(self.v_test)) * ufl.dx
        F += ufl.dot(vel, ufl.grad(self.phi)) * self.v_test * ufl.dx
        F -= self.source_func * self.v_test * ufl.dx
        
        # SUPG stabilization
        residual = (self.phi - self.phi_n)/dt + ufl.dot(vel, ufl.grad(self.phi))
        F += tau * ufl.dot(vel, ufl.grad(self.v_test)) * residual * ufl.dx
        
        # Create problem
        a = ufl.lhs(F)
        L = ufl.rhs(F)
        
        self.problem = LinearProblem(
            a, L, bcs=self.bcs,
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
                "ksp_rtol": 1e-6,
                "ksp_max_it": 1000
            },
            petsc_options_prefix="adr_"
        )
    
    def solve_timestep(self):
        """Solve timestep"""
        phi_new = self.problem.solve()
        self.phi_n.x.array[:] = phi_new.x.array[:]
        return phi_new
    
    def compute_mass(self):
        mass_form = fem.form(self.phi_n * ufl.dx)
        return fem.assemble_scalar(mass_form)
    
    def get_max_concentration(self):
        return float(np.max(self.phi_n.x.array[:]))
    
    def get_min_concentration(self):
        return float(np.min(self.phi_n.x.array[:]))

# Quality control

def check_plume_at_boundary(solver, params, threshold=1e-7):
    """Check if plume has reached domain boundary.
    
    Args:
        solver: ADRSolver2D instance
        params: SimulationParams
        threshold: Concentration threshold (kg/m³)
    
    Returns:
        bool: True if plume detected at boundary
    """
    coords = solver.V.tabulate_dof_coordinates()
    phi_values = solver.phi_n.x.array[:]  # Keep in kg/m³
    
    # Check concentrations near domain boundaries
    margin = params.Lx * 0.05  # 5% margin from edges
    
    # Near boundaries
    near_left = coords[:, 0] < margin
    near_right = coords[:, 0] > params.Lx - margin
    near_bottom = coords[:, 1] < margin
    near_top = coords[:, 1] > params.Ly - margin
    
    # Any point near boundary with concentration above threshold
    boundary_mask = near_left | near_right | near_bottom | near_top
    
    if boundary_mask.any():
        boundary_concentrations = phi_values[boundary_mask]
        max_boundary = np.max(boundary_concentrations)
        return max_boundary > threshold
    
    return False

def check_quality_metrics(solver, t, initial_mass):
    """Quality control - warnings only, never stops simulation"""
    mass = solver.compute_mass()  # kg
    max_phi = solver.get_max_concentration()  # kg/m³
    min_phi = solver.get_min_concentration()  # kg/m³
    
    metrics = {
        'time': float(t),
        'total_mass': float(mass),
        'max_concentration': float(max_phi),  # Keep in kg/m³
        'min_concentration': float(min_phi),  # Keep in kg/m³
    }
    
    issues = []
    
    # Check for negative concentrations
    if min_phi < -1e-10:  # -1e-10 kg/m³ = -0.1 µg/m³
        issues.append(f"Negative concentration: {min_phi:.2e} kg/m³")
    
    # Check for high concentrations (warning level)
    if max_phi > 1e-5:  # 1e-5 kg/m³ = 10,000 µg/m³
        issues.append(f"High concentration: {max_phi:.2e} kg/m³")
    
    # Check for very high concentrations (severe warning)
    if max_phi > 1e-3:  # 1e-3 kg/m³ = 1,000,000 µg/m³
        issues.append(f"VERY HIGH concentration: {max_phi:.2e} kg/m³")
    
    # Check for extreme concentrations (critical warning)
    if max_phi > 1e-2:  # 1e-2 kg/m³ = 10,000,000 µg/m³
        issues.append(f"EXTREME concentration: {max_phi:.2e} kg/m³")
    
    # Check for high total mass
    if mass > 1e5:
        issues.append(f"High total mass: {mass:.2e} kg")
    
    # Check for very high total mass
    if mass > 1e6:
        issues.append(f"VERY HIGH total mass: {mass:.2e} kg")
    
    # Check for extreme total mass
    if mass > 1e7:
        issues.append(f"EXTREME total mass: {mass:.2e} kg")
    
    metrics['issues'] = issues
    
    # NEVER raise exceptions - let simulation continue until boundary or T_max
    return metrics

# Data sampling

def save_partial_data(all_collocation_data, output_dir, log_file, snapshot_num, current_time, was_interrupted):
    """Save partial collocation data during execution"""
    if all_collocation_data:
        collocation_array = np.vstack(all_collocation_data)
        partial_file = output_dir / f"collocation_points_partial_{snapshot_num:04d}.npz"
        np.savez_compressed(partial_file, data=collocation_array)
        
        with open(log_file, 'a') as f:
            f.write(f"\n[PARTIAL SAVE] {len(collocation_array)} points saved at t={current_time:.1f}s ({snapshot_num} snapshots)\n")
            if was_interrupted:
                f.write(f"Save triggered by INTERRUPT\n")
        
        print(f"  ✓ Partial data saved: {len(collocation_array)} points to {partial_file.name}")
    else:
        with open(log_file, 'a') as f:
            f.write(f"\n[PARTIAL SAVE] No data to save at t={current_time:.1f}s\n")

def sample_collocation_points(solver, scenario, t, n_samples=1000):
    """Sample collocation points with Q_total and diameter as input features (11 columns)"""
    params = solver.params
    
    # Get mesh coordinates and solution values
    coords = solver.V.tabulate_dof_coordinates()
    phi_values = solver.phi_n.x.array[:]
    
    # Filter to interior points (avoid boundaries)
    margin = params.sampling_margin
    mask = (
        (coords[:, 0] > margin) & (coords[:, 0] < params.Lx - margin) &
        (coords[:, 1] > margin) & (coords[:, 1] < params.Ly - margin)
    )
    
    interior_coords = coords[mask]
    interior_phi = phi_values[mask]
    
    # Random sample from interior points
    n_available = len(interior_coords)
    if n_available == 0:
        return np.array([]).reshape(0, 11)
    
    n_actual = min(n_samples, n_available)
    indices = np.random.choice(n_available, size=n_actual, replace=False)
    
    samples = []
    for idx in indices:
        x_sample = interior_coords[idx, 0]
        y_sample = interior_coords[idx, 1]
        phi_val = interior_phi[idx]
        
        if np.isfinite(phi_val):
            samples.append([
                t, x_sample, y_sample,
                scenario['source_x'], scenario['source_y'],
                scenario['source_diameter'],  # Source diameter (m)
                scenario['Q_total'],  # Total mass rate (kg/s)
                scenario['wind_u'], scenario['wind_v'],
                scenario['D'],  # Diffusion coefficient
                phi_val
            ])
    
    return np.array(samples) if samples else np.array([]).reshape(0, 11)

def sample_initial_condition_points(solver, scenario, n_samples=5000):
    """IC points at t=0 with diameter and D as input features (11 columns)"""
    params = solver.params
    
    margin = params.sampling_margin
    x_min, x_max = margin, params.Lx - margin
    y_min, y_max = margin, params.Ly - margin
    
    samples = []
    
    for _ in range(n_samples):
        x_sample = np.random.uniform(x_min, x_max)
        y_sample = np.random.uniform(y_min, y_max)
        
        samples.append([
            0.0, x_sample, y_sample,
            scenario['source_x'], scenario['source_y'],
            scenario['source_diameter'],  # Source diameter (m)
            scenario['Q_total'],  # Total mass rate (kg/s)
            scenario['wind_u'], scenario['wind_v'],
            scenario['D'],  # Diffusion coefficient
            0.0  # phi=0 at t=0
        ])
    
    return np.array(samples)

# Visualization

def save_concentration_snapshot(solver, scenario, t, snapshot_num, output_dir):
    """Save heatmap visualization of concentration field"""
    
    # Get mesh coordinates and concentration values
    coords = solver.V.tabulate_dof_coordinates()
    phi_values = solver.phi_n.x.array[:]
    
    # Create triangulation for plotting
    x = coords[:, 0]
    y = coords[:, 1]
    triangulation = tri.Triangulation(x, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot concentration field with better color gradation
    max_phi = np.max(phi_values)
    min_phi = np.min(phi_values[phi_values > 0]) if np.any(phi_values > 0) else 0
    
    if max_phi > min_phi and min_phi > 0:
        # Use fewer, more distinct levels for better separation
        # Create levels that emphasize the plume structure
        n_levels = 20  # Fewer levels for better distinction
        
        # Create logarithmic levels but with better spacing
        log_min = np.log10(min_phi)
        log_max = np.log10(max_phi)
        levels = np.logspace(log_min, log_max, n_levels)
        
        # Add some intermediate levels for better gradation
        if n_levels > 10:
            # Add extra levels in the middle range where most variation occurs
            mid_log = (log_min + log_max) / 2
            extra_levels = np.logspace(mid_log - 0.5, mid_log + 0.5, 10)
            levels = np.sort(np.unique(np.concatenate([levels, extra_levels])))
        
        contourf = ax.tricontourf(triangulation, phi_values, levels=levels, 
                                  cmap='viridis', extend='both')
        
        # Add contour lines for better definition - fewer but more visible
        contour_lines = ax.tricontour(triangulation, phi_values, levels=8, 
                                   colors='black', alpha=0.4, linewidths=1.0)
    else:
        # Fallback to linear if no positive values
        levels = np.linspace(0, max(max_phi, 1e-10), 20)
        contourf = ax.tricontourf(triangulation, phi_values, levels=levels, 
                                  cmap='viridis', extend='max')
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(contourf, ax=ax, label='Concentration (kg/m³)', 
                       format='%.1e')  # Scientific notation
    
    # Mark source location
    ax.scatter(scenario['source_x'], scenario['source_y'], 
              c='black', marker='*', s=200, 
              label='Source', edgecolor='white', linewidth=1.5)
    
    # Add wind vector with better positioning
    wind_scale = 100  # Larger scale for visibility
    ax.arrow(100, 100, scenario['wind_u']*wind_scale, scenario['wind_v']*wind_scale,
            head_width=30, head_length=20, fc='blue', ec='blue', linewidth=2,
            label=f"Wind: {scenario['wind_speed']:.1f} m/s")
    
    # Labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f"Scenario {scenario['id']} | t={t:.1f}s | Q={scenario['Q_total']:.1f} kg/s\n"
                 f"Max φ={np.max(phi_values):.2e} kg/m³", fontsize=14)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / f"snapshot_{snapshot_num:03d}_t{int(t):04d}s.png", 
                dpi=150, bbox_inches='tight')
    plt.close(fig)

# Main runner

def run_single_scenario(scenario, base_output_dir, params):
    """Run one scenario with boundary-based stopping"""
    
    scenario_id = scenario['id']
    
    # Create output directory early
    output_dir = base_output_dir / f"scenario_{scenario_id:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive logging
    log_file = Path(f"scenario_{scenario_id:04d}_log.txt")
    
    # Global variables for interrupt handling
    global interrupted, all_collocation_data, history
    interrupted = False
    all_collocation_data = []
    history = []
    
    def signal_handler(sig, frame):
        global interrupted
        print(f"\n⚠ Interrupt received! Saving partial data for scenario {scenario_id}...")
        interrupted = True
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    with open(log_file, 'w') as f:
        f.write(f"Scenario {scenario_id} Log\n")
        f.write(f"{'='*50}\n")
        f.write(f"Source: ({scenario['source_x']:.1f}, {scenario['source_y']:.1f})\n")
        f.write(f"Source diameter: {scenario['source_diameter']:.0f} m\n")
        f.write(f"Q_total: {scenario['Q_total']:.3f} kg/s\n")
        f.write(f"Wind: {scenario['wind_speed']:.1f} m/s @ {scenario['wind_direction']:.0f}°\n")
        f.write(f"Wind components: u={scenario['wind_u']:.3f}, v={scenario['wind_v']:.3f} m/s\n")
        f.write(f"Stability: {scenario['stability_class']} (D={scenario['D']:.3e} m²/s)\n")
        f.write(f"Domain: {params.Lx:.0f}×{params.Ly:.0f} m, Mesh: {params.nx}×{params.ny}\n")
        f.write(f"{'='*50}\n\n")
    
    print(f"\n{'='*70}")
    print(f"SCENARIO {scenario_id}: 2D OPEN FIELD")
    print(f"Source: ({scenario['source_x']:.1f}, {scenario['source_y']:.1f})")
    print(f"Source diameter: {scenario['source_diameter']:.0f} m")
    print(f"Q_total: {scenario['Q_total']:.2f} kg/s")
    print(f"Wind: {scenario['wind_speed']:.1f} m/s @ {scenario['wind_direction']:.0f}°")
    print(f"Stability: {scenario['stability_class']} (D={scenario['D']:.2e} m²/s)")
    print(f"{'='*70}")
    
    # Create mesh
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0, 0], [params.Lx, params.Ly]],
        [params.nx, params.ny],
        mesh.CellType.quadrilateral
    )
    
    # Create source with diameter-based emission rate
    source = GaussianSource2D(
        scenario['source_x'], scenario['source_y'],
        scenario['Q_total'], scenario['source_diameter']
    )
    
    # Adaptive timestep per scenario (CFL ~ 0.5)
    grid_dx = params.Lx / params.nx
    wind_speed = max(0.1, float(scenario['wind_speed']))
    dt_advection = 0.5 * grid_dx / wind_speed
    dt_diffusion = 0.25 * grid_dx**2 / scenario['D']
    dt_scenario = min(dt_advection, dt_diffusion, 10.0)
    
    # Create solver
    params_scenario = SimulationParams(**vars(params))
    params_scenario.dt = dt_scenario
    solver = ADRSolver2D(params_scenario, domain, scenario, source)
    
    # Create snapshots directory for visualizations
    snapshots_dir = Path("data/snapshots") / f"scenario_{scenario_id:04d}"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    # Time loop with boundary detection and logging
    max_timesteps = int(params.T_max / dt_scenario)
    all_collocation_data = []
    
    # Sample and save IC data immediately
    ic_data = sample_initial_condition_points(solver, scenario)
    ic_file = output_dir / "ic_points.npz"
    np.savez_compressed(ic_file, data=ic_data)
    print(f"  Saved {len(ic_data)} IC points to {ic_file.name}")
    
    # Log IC data info
    with open(log_file, 'a') as f:
        f.write(f"IC Data: {len(ic_data)} points sampled\n")
        f.write(f"IC columns: t, x, y, source_x, source_y, source_diameter, Q_total, wind_u, wind_v, D, phi\n")
        if len(ic_data) > 0:
            f.write(f"IC sample range: t=[{ic_data[0,0]:.1f}, {ic_data[-1,0]:.1f}]\n")
            f.write(f"IC x range: [{ic_data[:,1].min():.1f}, {ic_data[:,1].max():.1f}]\n")
            f.write(f"IC y range: [{ic_data[:,2].min():.1f}, {ic_data[:,2].max():.1f}]\n")
            f.write(f"IC diameter range: [{ic_data[:,5].min():.1f}, {ic_data[:,5].max():.1f}] m\n")
        f.write(f"\n")
    
    # Track initial mass for monitoring
    initial_mass = solver.compute_mass()
    print(f"Initial mass: {initial_mass:.2e} kg\n")
    
    with open(log_file, 'a') as f:
        f.write(f"Initial mass: {initial_mass:.6e} kg\n")
        f.write(f"Adaptive dt: {dt_scenario:.6f} s\n")
        f.write(f"Grid spacing: {params.Lx/params.nx:.1f} m\n")
        f.write(f"\nTime evolution:\n")
        f.write(f"{'Step':<8} {'Time(s)':<10} {'Mass(kg)':<15} {'Max φ(kg/m³)':<18} {'Points':<8} {'Status':<15}\n")
    
    failed = False
    failure_message = ""
    
    try:
        snapshot_num = 0
        dt = dt_scenario
        snapshot_interval_s = 20.0  # Samples every 20s
        snapshot_interval_steps = max(1, int(round(snapshot_interval_s / dt)))
        next_snapshot_step = snapshot_interval_steps
        periodic_save_interval = 100  # Save partial data every 100 snapshots
        next_periodic_save = periodic_save_interval
        
        for n in range(max_timesteps):
            # Check for interrupt
            if interrupted:
                print(f"  Interrupt detected - stopping simulation and saving partial data...")
                break
                
            t = (n + 1) * dt
            solver.solve_timestep()
            
            if (n + 1) >= next_snapshot_step:
                # Quality check
                metrics = check_quality_metrics(solver, t, initial_mass)
                history.append(metrics)
                
                # Sample collocation data
                collocation_data = sample_collocation_points(solver, scenario, t)
                if len(collocation_data) > 0:
                    all_collocation_data.append(collocation_data)
                
                # Save visualization snapshot
                snapshot_num += 1
                save_concentration_snapshot(solver, scenario, t, snapshot_num, snapshots_dir)
                next_snapshot_step += snapshot_interval_steps
                
                # Log snapshot data
                with open(log_file, 'a') as f:
                    f.write(f"{snapshot_num:6d} {t:10.1f} {metrics['total_mass']:12.6e} {metrics['max_concentration']:15.6e} {len(collocation_data):8d}")
                    if len(metrics['issues']) > 0:
                        f.write(f" {' | '.join(metrics['issues'])}\n")
                    else:
                        f.write(f"\n")
                
                # Print progress
                if len(metrics['issues']) > 0:
                    print(f"[snap {snapshot_num:03d}] t={t:6.1f}s - Mass: {metrics['total_mass']:.2e}, "
                          f"Max φ: {metrics['max_concentration']:.2e} ⚠  {metrics['issues']}")
                else:
                    print(f"[snap {snapshot_num:03d}] t={t:6.1f}s - Mass: {metrics['total_mass']:.2e}, "
                          f"Max φ: {metrics['max_concentration']:.2e} ({len(collocation_data)} pts)")
                
                # Periodic partial data saving
                if snapshot_num >= next_periodic_save:
                    print(f"  [PERIODIC SAVE] Saving partial data after {snapshot_num} snapshots...")
                    save_partial_data(all_collocation_data, output_dir, log_file, snapshot_num, t, interrupted)
                    next_periodic_save += periodic_save_interval

                # Boundary detection check (after minimum time)
                if t >= params.T_min and n % params.boundary_check_interval == 0:
                    boundary_reached = check_plume_at_boundary(solver, params, params.boundary_threshold)
                    
                    # Log boundary check with details
                    with open(log_file, 'a') as f:
                        f.write(f"Boundary check at t={t:.1f}s: {boundary_reached} (threshold={params.boundary_threshold:.2e} kg/m³)\n")
                    
                    # Print boundary check status
                    if snapshot_num % 10 == 0:  # Print every 10th boundary check to avoid spam
                        print(f"  [BOUNDARY] t={t:.1f}s: plume at boundary? {boundary_reached}")
                    
                    if boundary_reached:
                        print(f"✓ Plume reached domain boundary at t={t:.0f}s - stopping simulation")
                        with open(log_file, 'a') as f:
                            f.write(f"STOPPED: Plume reached boundary at t={t:.1f}s\n")
                        break
    
    except RuntimeError as e:
        failed = True
        failure_message = str(e)
        print(f"\n⚠ Scenario {scenario_id} FAILED: {failure_message}")
        print(f"  Saving partial data ({len(all_collocation_data)} snapshots)...")
        save_partial_data(all_collocation_data, output_dir, log_file, snapshot_num, t if 't' in locals() else 0, False)
    
    except Exception as e:
        failed = True
        failure_message = f"Unexpected error: {str(e)}"
        print(f"\n⚠ Scenario {scenario_id} FAILED: {failure_message}")
        print(f"  Saving partial data ({len(all_collocation_data)} snapshots)...")
        save_partial_data(all_collocation_data, output_dir, log_file, snapshot_num, t if 't' in locals() else 0, False)
    
    # Handle interrupt case
    if interrupted:
        failed = True
        failure_message = "Simulation interrupted by user"
        save_partial_data(all_collocation_data, output_dir, log_file, snapshot_num, t if 't' in locals() else 0, True)
    
    # Save final summary to log
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"FINAL SUMMARY:\n")
        f.write(f"Total snapshots: {len(all_collocation_data)}\n")
        
        # Initialize collocation_array to avoid UnboundLocalError
        collocation_array = None
        if all_collocation_data:
            collocation_array = np.vstack(all_collocation_data)
        
        f.write(f"Total collocation points: {len(collocation_array) if collocation_array is not None else 0}\n")
        
        # Get final metrics safely
        final_metrics = history[-1] if history else {'total_mass': 0, 'max_concentration': 0}
        f.write(f"Final mass: {final_metrics['total_mass']:.6e} kg\n")
        f.write(f"Final max φ: {final_metrics['max_concentration']:.6e} kg/m³\n")
        f.write(f"Status: {'FAILED' if failed else 'COMPLETE'}\n")
        if failed:
            f.write(f"Error: {failure_message}\n")
        f.write(f"{'='*50}\n")
    
    # Save data (even if failed)
    # output_dir already defined at beginning of function
    
    # Debug: Check if we have collocation data
    print(f"  Saving data: {len(all_collocation_data)} snapshot arrays collected")
    
    if all_collocation_data:
        # collocation_array already defined above in final summary section
        collocation_file = output_dir / "collocation_points.npz"
        np.savez_compressed(collocation_file, data=collocation_array)
        print(f"  Saved {len(collocation_array)} collocation points to {collocation_file.name}")
        
        # Log collocation data info
        with open(log_file, 'a') as f:
            f.write(f"\nCollocation data saved: {len(collocation_array)} points\n")
            f.write(f"Columns: t, x, y, source_x, source_y, source_diameter, Q_total, wind_u, wind_v, D, phi\n")
            f.write(f"Time range: [{collocation_array[:,0].min():.1f}, {collocation_array[:,0].max():.1f}]\n")
            f.write(f"X range: [{collocation_array[:,1].min():.1f}, {collocation_array[:,1].max():.1f}]\n")
            f.write(f"Y range: [{collocation_array[:,2].min():.1f}, {collocation_array[:,2].max():.1f}]\n")
            f.write(f"Diameter range: [{collocation_array[:,5].min():.1f}, {collocation_array[:,5].max():.1f}] m\n")
            f.write(f"φ range: [{collocation_array[:,-1].min():.2e}, {collocation_array[:,-1].max():.2e}] kg/m³\n")
    else:
        print(f"  ⚠ No collocation data to save!")
        with open(log_file, 'a') as f:
            f.write(f"\nERROR: No collocation data saved!\n")
    
    ic_file = output_dir / "ic_points.npz"
    np.savez_compressed(ic_file, data=ic_data)
    print(f"  Saved {len(ic_data)} IC points to {ic_file.name}")
    
    if failed:
        print(f"[PARTIAL] Scenario {scenario_id} completed with PARTIAL data\n")
        return {'id': scenario_id, 'status': 'failed', 'error': failure_message, 'snapshots': len(all_collocation_data)}
    else:
        print(f"[COMPLETE] Scenario {scenario_id}\n")
        return {'id': scenario_id, 'status': 'complete', 'snapshots': len(all_collocation_data)}

# Main

def main():
    parser = argparse.ArgumentParser(description='2D FEniCS Data Generation for Benzene Dispersion')
    parser.add_argument('--scenario-id', type=int, default=None,
                       help='Run specific scenario ID (1-100). If not specified, runs all scenarios.')
    args = parser.parse_args()
    
    params = SimulationParams()
    base_output_dir = Path("data/simulations")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = generate_scenario_manifest(params)
    
    # Filter for specific scenario if requested
    if args.scenario_id is not None:
        scenarios = [s for s in scenarios if s['id'] == args.scenario_id]
        if not scenarios:
            print(f"Error: Scenario {args.scenario_id} not found!")
            return
        print(f"Running specific scenario {args.scenario_id}")
    
    results = []
    for scenario in scenarios:
        result = run_single_scenario(scenario, base_output_dir, params)
        results.append(result)
    
    # Summary
    complete = sum(1 for r in results if r['status'] == 'complete')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total scenarios: {len(results)}")
    print(f"  Completed: {complete}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed scenarios:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - Scenario {r['id']:04d}: {r['error']}")
        
        # Save failure log
        failure_log = base_output_dir / "failed_scenarios.json"
        with open(failure_log, 'w') as f:
            json.dump([r for r in results if r['status'] == 'failed'], f, indent=2)
        print(f"\nFailure details saved to: {failure_log}")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()