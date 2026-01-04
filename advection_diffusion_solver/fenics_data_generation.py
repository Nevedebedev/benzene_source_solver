"""
3D FEniCS Data Generation for Benzene Dispersion
Solves 3D advection-diffusion PDE with stack height effects
Samples ground-level (z=0) concentrations for PINN training
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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import multiprocessing
from multiprocessing import Pool, cpu_count
import os
import sys
import argparse
import fcntl

from atmospheric_physics import (
    sample_meteorological_conditions,
    get_pasquill_class,
    stability_class_to_D,
    get_vertical_velocity,
    check_steady_state_mass
)


# Configuration

@dataclass
class SimulationParams:
    Lx: float = 40000.0   # 40km domain
    Ly: float = 40000.0   # 40km domain
    Lz: float = 5.0        # 5m height (ground-level modeling)
    nx: int = 200         # 200m cells (40km/200)
    ny: int = 200         # 200m cells
    nz: int = 5           # 1m vertical cells
    # Total: 200×200×5 = 200,000 cells (manageable!)
    T: float = 500.0           # Total simulation time (s)
    dt: float = 0.05           # Time step (s) - reduced for stability
    output_freq: int = 100     # Save every N timesteps (every 5s)
    # D is now DYNAMIC - calculated from stability class per scenario
    z0: float = 0.03           # Surface roughness (m)
    z: float = 2.0             # Release height (m)
    
    # Emission rates - REALISTIC total mass flux (ambient monitoring)
    Q_total_min: float = 5.0   # Small facility (kg/s)
    Q_total_max: float = 100.0 # Large refinery (kg/s)
    Q_log_scale: bool = True   # Log-uniform: more small/medium facilities

    @property
    def sigma_xy(self):
        """Horizontal source spread - adaptive to mesh resolution"""
        cell_size_xy = self.Lx / self.nx
        return max(50.0, 2.5 * cell_size_xy)

    @property
    def sigma_z(self):
        """Vertical source spread - SMALL for ground-level sources in shallow domain"""
        # For 5m domain with ground-level sources, use very tight vertical spread
        # This ensures most of the Gaussian is captured within [0, 5m]
        return 0.5  # Fixed at 0.5m for ground-level sources
    supg_factor: float = 0.5   # SUPG stabilization factor
    sampling_margin_xy: float = 500.0
    sampling_margin_z: float = 50.0

    # Time control: run until steady state OR cap
    T_min: float = 2000.0   # Minimum runtime before checking steady state
    T_max: float = 10000.0  # Maximum runtime (safety cap)

    # Steady state detection
    steady_state_tolerance: float = 1e-4  # Relative change tolerance
    steady_state_window: int = 5          # Number of recent snapshots to check
    steady_state_buffer: float = 500.0    # Continue this long after detection (s)

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
    """10 strategic source positions scaled to domain size."""
    Lx, Ly = params.Lx, params.Ly
    cx, cy = Lx / 2.0, Ly / 2.0
    ox, oy = 0.1 * Lx, 0.1 * Ly
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
        SourceLocation(10, cx, cy + 0.8 * oy, "N_inner"),
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
    """400 scenarios with RANDOM Q₀ for realistic emission rate training"""
    scenarios = []
    scenario_id = 0
    
    sources = generate_source_locations(params)
    
    # 10 strategic wind conditions covering parameter space
    winds = [
        WindCondition(0.5, 0),      # Calm easterly
        WindCondition(0.5, 180),    # Calm westerly
        WindCondition(2.0, 45),     # Light NE (common in Houston)
        WindCondition(2.0, 225),    # Light SW (common in Houston)
        WindCondition(4.0, 90),     # Moderate northerly
        WindCondition(4.0, 270),    # Moderate southerly (Gulf breeze)
        WindCondition(6.0, 135),    # Strong SE (storm approach)
        WindCondition(6.0, 315),    # Strong NW (cold front)
        WindCondition(10.0, 0),     # Very strong easterly
        WindCondition(10.0, 180),   # Very strong westerly
    ]
    
    # Random generator (reproducible)
    rng = np.random.default_rng(seed=42)

    # Generate all combinations with random Q_total AND DYNAMIC D
    for source in sources:
        for wind in winds:
            # Random diameter for this source
            d_min, d_max = 150.0, 4500.0
            diameter = rng.uniform(d_min, d_max)
            
            # Get emission rate based on diameter category
            Q_total = get_emission_rate_for_diameter(diameter, rng)
            
            scenario_id += 1
            
            # Sample meteorology and calculate D from stability class
            solar, cloud, hour, is_day = sample_meteorological_conditions(
                wind.speed, rng
            )
            
            stability_class = get_pasquill_class(
                wind.speed, solar, cloud, is_day
            )
            
            D_horizontal, D_vertical = stability_class_to_D(
                stability_class, wind.speed, params.z0
            )

            # Calculate vertical velocity from stability class
            wind_w = get_vertical_velocity(stability_class, wind.speed)
             
            scenarios.append({
                'id': scenario_id,
                'geometry': 'open_field',
                'source_id': source.id,
                'source_x': source.x,
                'source_y': source.y,
                'source_name': source.name,
                'source_diameter': float(diameter),  # Add diameter
                # source_z removed (always 0)
                'Q_total': float(Q_total),  # Total mass rate (kg/s)
                'wind_speed': wind.speed,
                'wind_direction': wind.direction,
                'wind_u': wind.u,
                'wind_v': wind.v,
                'wind_w': float(wind_w),
                'D': float(D_horizontal),
                'D_horizontal': float(D_horizontal),
                'D_vertical': float(D_vertical),
                'stability_class': stability_class,
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
    if small_scenarios > 0:
        small_q = [s['Q_total'] for s in scenarios if s['source_diameter'] < 1500]
        print(f"    Q range: {min(small_q):.2f}-{max(small_q):.2f} kg/s")
    print(f"  Medium facilities (1500-2500m): {medium_scenarios} scenarios")
    if medium_scenarios > 0:
        medium_q = [s['Q_total'] for s in scenarios if 1500 <= s['source_diameter'] < 2500]
        print(f"    Q range: {min(medium_q):.2f}-{max(medium_q):.2f} kg/s")
    print(f"  Large facilities (2500-4500m): {large_scenarios} scenarios")
    if large_scenarios > 0:
        large_q = [s['Q_total'] for s in scenarios if s['source_diameter'] >= 2500]
        print(f"    Q range: {min(large_q):.2f}-{max(large_q):.2f} kg/s")
    print(f"  Stability classes: A-F (D derived from stability)")
    print(f"  Distribution: Neutral-biased (realistic)")
    print(f"  Q_total: {params.Q_total_min:.1f} - {params.Q_total_max:.1f} kg/s")
    print(f"  D_h range: 1.0 - 100.0 m²/s | D_v range: 0.1 - 50.0 m²/s (calculated from stability)")
    print(f"  Distribution: Neutral-biased (realistic)")
    return scenarios

# Boundary conditions

def apply_boundary_conditions_3D(V, params):
    """
    3D boundary conditions - natural BC on all boundaries.
    Removes ground Dirichlet BC to allow realistic ground-level concentrations.
    Lateral boundaries handled by explicit outflow term in weak form.
    """
    def ground_boundary(x):
        return np.isclose(x[2], 0.0)

    # ground_dofs = fem.locate_dofs_geometrical(V, ground_boundary)
    # bc_ground = fem.dirichletbc(PETSc.ScalarType(0), ground_dofs, V)

    # No boundary conditions - natural BC everywhere
    return []

# Source

class GaussianSource3D:
    """3D Gaussian emission source at height H."""
    def __init__(self, x_source, y_source, z_source, Q_total, sigma_xy, sigma_z):
        self.xs = x_source
        self.ys = y_source
        self.zs = z_source
        self.Q_total = Q_total
        self.sigma_xy = sigma_xy
        self.sigma_z = sigma_z

        self.Q0 = Q_total / ((2 * np.pi)**1.5 * sigma_xy**2 * sigma_z)

    def __call__(self, x):
        r_xy_squared = (x[0] - self.xs)**2 + (x[1] - self.ys)**2
        r_z_squared = (x[2] - self.zs)**2
        gaussian = np.exp(
            -r_xy_squared / (2 * self.sigma_xy**2)
            -r_z_squared / (2 * self.sigma_z**2)
        )
        return self.Q0 * gaussian

# ADR Solver

class ADRSolver:
    """Advection-Diffusion-Reaction solver with DYNAMIC D"""
    
    def __init__(self, params, domain, scenario, source):
        self.params = params
        self.domain = domain
        self.wind_u = scenario['wind_u']
        self.wind_v = scenario['wind_v']
        self.wind_w = scenario['wind_w']
        self.D_h = scenario['D_horizontal']
        self.D_v = scenario['D_vertical']
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

            # DIAGNOSTIC: Verify source was captured properly
            source_integral = fem.assemble_scalar(fem.form(self.source_func * ufl.dx))
            source_integral = self.domain.comm.allreduce(source_integral, op=MPI.SUM)
            capture_efficiency = (source_integral / scenario['Q_total']) * 100

            print(f"  === SOURCE DIAGNOSTICS ===")
            print(f"  Wind: u={self.wind_u:.3f}, v={self.wind_v:.3f}, w={self.wind_w:.3f} m/s")
            print(f"  Source integral: {source_integral:.6e} kg/s")
            print(f"  Target Q_total: {scenario['Q_total']:.6e} kg/s")
            print(f"  Capture efficiency: {capture_efficiency:.2f}%")

            # Check source at its center point (z=1m where source is centered)
            source_center_value = self.source([scenario['source_x'], scenario['source_y'], 1.0])
            print(f"  Source at center ({scenario['source_x']}, {scenario['source_y']}, 1.0m): {source_center_value:.6e}")

            # Check max value in source function
            source_max = self.source_func.x.array[:].max()
            print(f"  Max source in mesh: {source_max:.6e}")
            print(f"  Mesh cell height: {self.params.Lz/self.params.nz:.2f}m")
            print(f"  Source height: 0.0m (ground level - automatic)")
            print(f"  Sigma_z: {self.params.sigma_z:.2f}m")
            print(f"  === END DIAGNOSTICS ===")

            if capture_efficiency < 50:
                print(f"  WARNING: Only capturing {capture_efficiency:.1f}% of source!")
                print(f"  This will result in zero or near-zero concentrations!")
        except Exception as e:
            raise RuntimeError(f"Source interpolation failed: {str(e)}")
        
        # Boundary conditions (wind-aware inflow/outflow)
        self.bcs = apply_boundary_conditions_3D(self.V, params)
        
        # Setup problem
        self._setup_variational_problem()
    
    def _setup_variational_problem(self):
        """Setup ADR weak form WITH explicit outflow boundary term AND DYNAMIC D"""
        dt = self.params.dt
        D_h = self.D_h
        D_v = self.D_v
        u = self.wind_u
        v = self.wind_v
        
        # Velocity - set w=0 for stable simulation
        vel = ufl.as_vector([u, v, 0.0])  # Set w=0.0
        vel_mag = ufl.sqrt(u**2 + v**2) + 1e-10
        
        # SUPG parameter
        h = ufl.CellDiameter(self.domain)
        tau = self.params.supg_factor * h / (2 * vel_mag)
        
        # Standard volume terms
        F = (self.phi - self.phi_n)/dt * self.v_test * ufl.dx
        D_tensor = ufl.as_matrix([
            [D_h, 0, 0],
            [0, D_h, 0],
            [0, 0, D_v]
        ])
        F += ufl.inner(D_tensor * ufl.grad(self.phi), ufl.grad(self.v_test)) * ufl.dx
        F += ufl.dot(vel, ufl.grad(self.phi)) * self.v_test * ufl.dx
        
        # Source term (RHS)
        F -= self.source_func * self.v_test * ufl.dx
        
        # Natural boundary conditions - NO explicit boundary term needed!
        # The weak form automatically handles outflow naturally
        # This was the approach that worked in the successful run
        
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
        """Solve timestep - SUPG prevents negatives"""
        phi_new = self.problem.solve()
        # No clipping needed with SUPG
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

def check_quality_metrics(solver, t, initial_mass):
    """Quality control with SUPG stabilization"""
    mass = solver.compute_mass()
    max_phi = solver.get_max_concentration()
    min_phi = solver.get_min_concentration()
    
    metrics = {
        'time': float(t),
        'total_mass': float(mass),
        'max_concentration': float(max_phi),
        'min_concentration': float(min_phi),
    }
    
    issues = []
    
    # SUPG allows small numerical undershoots - only warn if significant
    if min_phi < -0.001:  # More than 1 mg/m³ negative
        issues.append(f"Negative concentration: {min_phi:.2e} kg/m³")
    
    # Realistic max for Q_total ∈ [5, 100] kg/s
    if max_phi > 200:  # Adjusted for SUPG + natural outflow
        issues.append(f"High concentration: {max_phi:.2e} kg/m³")
    
    # Mass should increase then stabilize
    if mass > 1e5:  # Adjusted for new emission rates
        issues.append(f"High total mass: {mass:.2e} kg")

    coords = solver.V.tabulate_dof_coordinates()
    phi_vals = solver.phi_n.x.array[:]
    high_altitude_mask = (coords[:, 2] > 300.0)
    if high_altitude_mask.any():
        max_high = float(np.max(phi_vals[high_altitude_mask]))
        if max_high > 10.0:
            issues.append(f"High altitude concentration: {max_high:.2e} kg/m³")
    
    metrics['issues'] = issues
    
    # CRITICAL: Stop if truly catastrophic
    if min_phi < -0.1 or max_phi > 1000:  # Relaxed from -0.01
        raise RuntimeError(f"Simulation unstable at t={t:.1f}s - STOPPING")
    
    return metrics

# Steady-state detection
def check_steady_state(history, tolerance, window):
    """Return True if recent max concentration change is below tolerance."""
    if len(history) < window:
        return False
    recent_max = [h['max_concentration'] for h in history[-window:]]
    mean_max = float(np.mean(recent_max))
    if mean_max == 0.0 or not np.isfinite(mean_max):
        return False
    max_change = (max(recent_max) - min(recent_max)) / mean_max
    return max_change < tolerance

# Data sampling

def sample_collocation_points(solver, scenario, t, n_samples=1000):
    """
    Sample ground-level collocation points from z=1m mesh layer (13 columns).
    
    All sources at z=1m, sampling at z=1m to match source center.
    This avoids interpolation issues by using actual mesh node values.
    """
    params = solver.params
    
    coords = solver.V.tabulate_dof_coordinates()
    phi_values = solver.phi_n.x.array[:]
    
    z_target = 1.0  # Back to z=1.0m
    z_tolerance = 0.5
    height_mask = np.abs(coords[:, 2] - z_target) < z_tolerance
     
    margin_xy = params.sampling_margin_xy
    interior_mask = (
        (coords[:, 0] > margin_xy) & (coords[:, 0] < params.Lx - margin_xy) &
        (coords[:, 1] > margin_xy) & (coords[:, 1] < params.Ly - margin_xy)
    )
     
    mask = height_mask & interior_mask
    interior_coords = coords[mask]
    interior_phi = phi_values[mask]
     
    finite_mask = np.isfinite(interior_phi) & (interior_phi >= 0)
    interior_coords = interior_coords[finite_mask]
    interior_phi = interior_phi[finite_mask]
     
    n_available = len(interior_coords)
    if n_available == 0:
        print(f"    ⚠️  No nodes found at z≈{z_target}m")
        return np.array([]).reshape(0, 13)
     
    n_actual = min(n_samples, n_available)
    indices = np.random.choice(n_available, size=n_actual, replace=False)
     
    stability_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    stability_encoded = stability_map.get(scenario['stability_class'], 3)
     
    samples = []
    for idx in indices:
        x_sample = float(interior_coords[idx, 0])
        y_sample = float(interior_coords[idx, 1])
        phi_val = float(interior_phi[idx])
         
        samples.append([
            t, x_sample, y_sample, 1.0,  # z=1.0m
            scenario['source_x'], scenario['source_y'],
            scenario['source_diameter'],
            scenario['Q_total'],
            scenario['wind_u'], scenario['wind_v'],
            stability_encoded,  # Use stability class instead of D
            0.0,  # vertical_velocity=0 (no stack height)
            phi_val
        ])
     
    return np.array(samples) if samples else np.array([]).reshape(0, 13)



def sample_initial_condition_points(solver, scenario, n_samples=5000):
    """IC points at t=0 (13 columns). All points at z=1m to match source."""
    params = solver.params
    
    margin_xy = params.sampling_margin_xy
    x_min, x_max = margin_xy, params.Lx - margin_xy
    y_min, y_max = margin_xy, params.Ly - margin_xy
    
    # Encode stability class as numeric
    stability_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    stability_encoded = stability_map.get(scenario['stability_class'], 3)
    
    samples = []
    
    for _ in range(n_samples):
        x_sample = np.random.uniform(x_min, x_max)
        y_sample = np.random.uniform(y_min, y_max)
        
        samples.append([
            0.0, x_sample, y_sample, 1.0,  # t=0, z=1.0m
            scenario['source_x'], scenario['source_y'],
            scenario['source_diameter'],  # Add diameter
            scenario['Q_total'],
            scenario['wind_u'], scenario['wind_v'],
            stability_encoded,  # Use stability class instead of D
            0.0,  # vertical_velocity=0
            0.0  # phi=0 at t=0
        ])
    
    return np.array(samples)

# Visualization

def save_concentration_snapshot(solver, scenario, t, snapshot_num, output_dir):
    """Save heatmap visualization of concentration field"""
    try:
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get mesh coordinates and concentration values
        coords = solver.V.tabulate_dof_coordinates()
        phi_values = solver.phi_n.x.array[:]

        ground_mask = (coords[:, 2] < 5.0)
        coords = coords[ground_mask]
        phi_values = phi_values[ground_mask]

        finite_mask = np.isfinite(phi_values)
        coords = coords[finite_mask]
        phi_values = phi_values[finite_mask]
        
        if len(coords) == 0 or len(phi_values) == 0:
            print(f"  ⚠ No ground-level data to plot at t={t:.1f}s")
            return
            
        # Create triangulation for plotting
        x = coords[:, 0]
        y = coords[:, 1]
        triangulation = tri.Triangulation(x, y)
        
        # Create figure with tight layout
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot concentration field with log/linear scaling
        max_phi = float(np.nanmax(phi_values)) if phi_values.size else 0.0
        pos = phi_values[phi_values > 0.0]
        min_phi = float(np.min(pos)) if pos.size else 0.0

        contour = None
        try:
            if max_phi > min_phi and min_phi > 0.0 and np.isfinite(max_phi) and np.isfinite(min_phi):
                levels = np.logspace(np.log10(min_phi), np.log10(max_phi), 50)
            else:
                vmax = max(max_phi if np.isfinite(max_phi) else 0.0, 1e-10)
                levels = np.linspace(0.0, vmax, 50)

            contour = ax.tricontourf(triangulation, phi_values, levels=levels, cmap='YlOrRd')
        except Exception:
            contour = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=phi_values,
                s=6,
                cmap='YlOrRd',
                vmin=0.0,
                vmax=max(max_phi if np.isfinite(max_phi) else 0.0, 1e-10),
            )
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, label='Concentration (kg/m³)')
        
        # Mark source location
        ax.scatter(scenario['source_x'], scenario['source_y'], 
                  c='black', marker='*', s=200, 
                  label='Source', edgecolor='white', linewidth=1.5)
        
        # Add wind vector (scaled for visibility)
        wind_scale = 100  # Arrow length
        ax.arrow(100, 100, scenario['wind_u']*wind_scale, scenario['wind_v']*wind_scale,
                head_width=30, head_length=20, fc='blue', ec='blue', linewidth=2,
                label=f"Wind: {scenario['wind_speed']:.1f} m/s")
        
        # Set plot properties
        ax.set_title(f"Scenario {scenario['id']} | t={t:.1f}s | Q={scenario['Q_total']:.1f} kg/s\n"
                    f"Max φ={max_phi:.2e} kg/m³", fontsize=14)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the figure
        output_path = output_dir / f'snapshot_{snapshot_num:03d}_t{int(t):04d}s.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Saved snapshot {snapshot_num:03d} at t={t:.1f}s")
        
    except Exception as e:
        print(f"  ⚠ Error saving snapshot at t={t:.1f}s: {str(e)}")
        import traceback
        traceback.print_exc()


# Main runner

def run_single_scenario(scenario, base_output_dir, params):
    """Run one scenario (open field)"""
    
    scenario_id = scenario['id']
    log_path = Path("generation_log.txt")

    def log_line(line: str):
        """Thread-safe logging with file locking."""
        try:
            with open(log_path, "a") as f:
                # Lock the file for exclusive write access
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line.rstrip("\n") + "\n")
                    f.flush()  # Ensure data is written
                finally:
                    # Unlock the file
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (IOError, OSError) as e:
            # If logging fails, print to stderr but don't crash
            print(f"Warning: Failed to write to log file: {e}", file=sys.stderr)
    
    print(f"\n{'='*70}")
    print(f"SCENARIO {scenario_id}: OPEN FIELD")
    print(f"Source: ({scenario['source_x']:.1f}, {scenario['source_y']:.1f})")
    print(f"Stack height: 0.0 m (ground level - automatic)")
    print(f"Q_total: {scenario['Q_total']:.2f} kg/s")
    print(f"Wind: {scenario['wind_speed']:.1f} m/s @ {scenario['wind_direction']:.0f}°")
    print(f"Stability: {scenario['stability_class']} (D_h={scenario['D_horizontal']:.2e}, D_v={scenario['D_vertical']:.2e} m²/s)")
    print(f"Conditions: Solar={scenario['solar_radiation']:.0f} W/m², Cloud={scenario['cloud_cover']:.0f}%")
    print(f"{'='*70}")

    log_line(f"[SCENARIO {scenario_id:04d}] Source=({scenario['source_x']:.1f},{scenario['source_y']:.1f}) H=0.0m(auto) Q={scenario['Q_total']:.3f}kg/s Wind={scenario['wind_speed']:.2f}m/s@{scenario['wind_direction']:.0f} Stability={scenario['stability_class']}")
    
    # Create mesh
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [[0, 0, 0], [params.Lx, params.Ly, params.Lz]],
        [params.nx, params.ny, params.nz],
        mesh.CellType.hexahedron
    )
    
    # Create source with scenario-specific Q_total
    source = GaussianSource3D(
        scenario['source_x'],
        scenario['source_y'],
        1.0,  # Back to z=1.0m
        scenario['Q_total'],
        scenario['source_diameter'] / 4.0,  # Calculate sigma_xy from diameter
        0.5  # Fixed sigma_z for ground-level
    )

    grid_dx = params.Lx / params.nx
    grid_dz = params.Lz / params.nz
    wind_speed = max(0.1, float(scenario['wind_speed']))
    dt_advection = 0.5 * grid_dx / wind_speed
    dt_diffusion_h = 0.25 * grid_dx**2 / scenario['D_horizontal']
    dt_diffusion_v = 0.25 * grid_dz**2 / scenario['D_vertical']
    dt_scenario = min(dt_advection, dt_diffusion_h, dt_diffusion_v, 10.0)
    log_line(f"  dt={dt_scenario:.6g}s dt_adv={dt_advection:.6g} dt_diff_h={dt_diffusion_h:.6g} dt_diff_v={dt_diffusion_v:.6g}")

    # Create solver (pass entire scenario for dynamic D)
    params_scenario = SimulationParams(**vars(params))
    params_scenario.dt = dt_scenario
    solver = ADRSolver(params_scenario, domain, scenario, source)

    # Verify source integration (using solver's interpolated source)
    source_integral = fem.assemble_scalar(fem.form(solver.source_func * ufl.dx))
    source_integral = domain.comm.allreduce(source_integral, op=MPI.SUM)
    capture_efficiency = (source_integral / scenario['Q_total']) * 100
    log_line(f"  source_integral={source_integral:.6e}kg/s capture_efficiency={capture_efficiency:.2f}% sigma_xy={params.sigma_xy:.3f} sigma_z={params.sigma_z:.3f}")
    
    # Quick source test
    solver.solve_timestep()
    test_mass = solver.compute_mass()
    if test_mass < 1e-10:  # Very small mass suggests source isn't working
        raise RuntimeError(f"Source test failed - no mass detected after first step")
    solver.phi_n.x.array[:] = 0.0  # Reset for actual simulation
    
    # Create snapshots directory for visualizations
    snapshots_dir = Path("data/snapshots") / f"scenario_{scenario_id:04d}"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save initial snapshot at t=0
    snapshot_num = 0
    save_concentration_snapshot(solver, scenario, 0, snapshot_num, snapshots_dir)
    print(f"✓ Saved initial snapshot at t=0s")
    log_line(f"  snapshot=000 t=0.0s saved={str((snapshots_dir / f'snapshot_{snapshot_num:03d}_t{int(0):04d}s.png').resolve())}")
    
    # Time loop with error handling
    max_timesteps = int(params.T_max / dt_scenario)
    all_collocation_data = []
    
    ic_data = sample_initial_condition_points(solver, scenario)
    
    # Track initial mass for monitoring
    initial_mass = solver.compute_mass()
    
    failed = False
    failure_message = ""
    
    try:
        history = []
        steady_state_reached = False
        steady_state_time = None
        snapshot_num = 0
        dt = dt_scenario
        snapshot_interval_s = 20.0
        snapshot_interval_steps = max(1, int(round(snapshot_interval_s / dt)))
        next_snapshot_step = snapshot_interval_steps
        for n in range(max_timesteps):
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
                
                # Print progress
                if len(metrics['issues']) > 0:
                    print(f"[snap {snapshot_num:03d}] t={t:6.1f}s - Mass: {metrics['total_mass']:.2e}, "
                          f"Max φ: {metrics['max_concentration']:.2e} ⚠  {metrics['issues']}")
                else:
                    print(f"[snap {snapshot_num:03d}] t={t:6.1f}s - Mass: {metrics['total_mass']:.2e}, "
                          f"Max φ: {metrics['max_concentration']:.2e} ({len(collocation_data)} pts)")

                log_line(f"  [snap {snapshot_num:03d}] t={t:.3f}s mass={metrics['total_mass']:.6e} max={metrics['max_concentration']:.6e} min={metrics['min_concentration']:.6e} pts={len(collocation_data)} issues={';'.join(metrics['issues']) if metrics['issues'] else 'none'}")

                # Steady-state check (after minimum time) - using mass-based detection
                if t >= params.T_min:
                    if check_steady_state_mass(
                        history, 
                        scenario['Q_total'], 
                        dt_scenario,
                        params.steady_state_tolerance, 
                        params.steady_state_window
                    ):
                        steady_state_reached = True
                        steady_state_time = t
                        print(f"✓ Steady state reached at t={t:.0f}s (mass balance) — continuing for buffer {params.steady_state_buffer:.0f}s")

                        # Continue for steady-state buffer duration
                        continue_steps = int(params.steady_state_buffer / dt)
                        for m in range(continue_steps):
                            t_buf = t + (m + 1) * dt
                            solver.solve_timestep()
                            if (m + 1) % params.output_freq == 0:
                                metrics_buf = check_quality_metrics(solver, t_buf, initial_mass)
                                history.append(metrics_buf)
                                collocation_data_buf = sample_collocation_points(solver, scenario, t_buf)
                                if len(collocation_data_buf) > 0:
                                    all_collocation_data.append(collocation_data_buf)
                                snapshot_num += 1
                                save_concentration_snapshot(solver, scenario, t_buf, snapshot_num, snapshots_dir)
                                if len(metrics_buf['issues']) > 0:
                                    print(f"[snap {snapshot_num:03d}] t={t_buf:6.1f}s - Mass: {metrics_buf['total_mass']:.2e}, "
                                          f"Max φ: {metrics_buf['max_concentration']:.2e} ⚠  {metrics_buf['issues']}")
                                else:
                                    print(f"[snap {snapshot_num:03d}] t={t_buf:6.1f}s - Mass: {metrics_buf['total_mass']:.2e}, "
                                          f"Max φ: {metrics_buf['max_concentration']:.2e} ({len(collocation_data_buf)} pts)")
                        print(f"Completed at steady state: t={steady_state_time:.0f}s")
                        log_line(f"  steady_state_reached t={steady_state_time:.6g}s")
                        break

    except KeyboardInterrupt:
        failed = True
        failure_message = "Interrupted by user"
        print(f"\n⚠ Scenario {scenario_id} INTERRUPTED: saving partial outputs...")
        log_line(f"  INTERRUPTED")
    
    except RuntimeError as e:
        failed = True
        failure_message = str(e)
        print(f"\n⚠ Scenario {scenario_id} FAILED: {failure_message}")
        print(f"  Saving partial data ({len(all_collocation_data)} snapshots)...")
        log_line(f"  FAILED error={failure_message}")
    
    except Exception as e:
        failed = True
        failure_message = f"Unexpected error: {str(e)}"
        print(f"\n⚠ Scenario {scenario_id} FAILED: {failure_message}")
        print(f"  Saving partial data ({len(all_collocation_data)} snapshots)...")
        log_line(f"  FAILED error={failure_message}")
    
    # Save data (even if failed)
    output_dir = base_output_dir / f"scenario_{scenario_id:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Saving data: {len(all_collocation_data)} snapshot arrays collected")
    
    if all_collocation_data:
        collocation_array = np.vstack(all_collocation_data)
        collocation_file = output_dir / "collocation_points.npz"
        np.savez_compressed(collocation_file, data=collocation_array)
        print(f"  Saved {len(collocation_array)} collocation points to {collocation_file.name}")
        log_line(f"  saved_collocation_points={len(collocation_array)} file={str(collocation_file.resolve())}")
    else:
        print(f"  ⚠ No collocation data to save!")
        log_line(f"  saved_collocation_points=0")
    
    ic_file = output_dir / "ic_points.npz"
    np.savez_compressed(ic_file, data=ic_data)
    print(f"  Saved {len(ic_data)} IC points to {ic_file.name}")
    log_line(f"  saved_ic_points={len(ic_data)} file={str(ic_file.resolve())}")
    
    if failed:
        print(f"[PARTIAL] Scenario {scenario_id} completed with PARTIAL data\n")
        log_line(f"  STATUS=partial snapshots={len(all_collocation_data)}")
        return {'id': scenario_id, 'status': 'failed', 'error': failure_message, 'snapshots': len(all_collocation_data)}
    else:
        print(f"[COMPLETE] Scenario {scenario_id}\n")
        log_line(f"  STATUS=complete snapshots={len(all_collocation_data)}")
        return {'id': scenario_id, 'status': 'complete', 'snapshots': len(all_collocation_data)}

# Main

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='FEniCS 3D Advection-Diffusion Data Generation for PINN Training'
    )
    parser.add_argument(
        '--parallel', '--speed',
        action='store_true',
        help='Enable parallel processing mode'
    )
    parser.add_argument(
        '--cores',
        type=int,
        default=None,
        help='Number of cores to use for parallel processing (default: CPU count - 4)'
    )
    args = parser.parse_args()
    
    params = SimulationParams()
    base_output_dir = Path("data/simulations")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = generate_scenario_manifest(params)
    
    # Determine execution mode and number of cores
    use_parallel = args.parallel
    if use_parallel:
        # Determine number of cores
        if args.cores is not None:
            n_cores = args.cores
        else:
            n_cores = int(os.getenv('FENICS_CORES', cpu_count() - 4))
        
        # Clamp to reasonable values
        n_cores = max(1, min(n_cores, cpu_count()))
        
        print(f"\n{'='*70}")
        print(f"PARALLEL EXECUTION MODE")
        print(f"{'='*70}")
        print(f"Total scenarios: {len(scenarios)}")
        print(f"Using {n_cores} CPU cores (out of {cpu_count()} available)")
        print(f"{'='*70}\n")
        
        start_time = datetime.now()
        
        # Parallel execution using multiprocessing Pool
        with Pool(n_cores) as pool:
            results = pool.starmap(
                run_single_scenario,
                [(scenario, base_output_dir, params) for scenario in scenarios]
            )
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
    else:
        # Sequential execution (original behavior)
        print(f"\n{'='*70}")
        print(f"SEQUENTIAL EXECUTION MODE")
        print(f"{'='*70}")
        print(f"Total scenarios: {len(scenarios)}")
        print(f"{'='*70}\n")
        
        start_time = datetime.now()
        
        results = []
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[Progress: {i}/{len(scenarios)}]")
            result = run_single_scenario(scenario, base_output_dir, params)
            results.append(result)
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
    
    # Summary
    complete = sum(1 for r in results if r['status'] == 'complete')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total scenarios: {len(results)}")
    print(f"  Completed: {complete}")
    print(f"  Failed: {failed}")
    print(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    if use_parallel:
        print(f"Average time per scenario: {elapsed/len(scenarios)/60:.1f} minutes")
    
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