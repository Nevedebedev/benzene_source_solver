"""
Atmospheric physics calculations for stability and diffusion
Used in BOTH training data generation AND deployment
"""
import numpy as np

def stability_class_to_D(stability_class: str, 
                         wind_speed: float,
                         z0: float = 0.03) -> tuple:
    """Return (D_horizontal, D_vertical) in m^2/s.

    Uses a Pasquill-Gifford stability-dependent parameterization inspired by
    EPA AERMOD formulations (EPA-454/B-03-001) and standard surface-layer
    similarity theory for friction velocity.
    """
    kappa = 0.41
    z_ref = 10.0
    stability_class = normalize_stability_class(stability_class)
    u_star = (kappa * float(wind_speed)) / np.log(z_ref / float(z0))

    L_values = {
        'A': -10.0,
        'B': -25.0,
        'C': -50.0,
        'D': 1e6,
        'E': 50.0,
        'F': 20.0
    }
    L = float(L_values[stability_class])

    h_mix_values = {
        'A': 1500.0,
        'B': 1200.0,
        'C': 900.0,
        'D': 800.0,
        'E': 400.0,
        'F': 200.0
    }
    h_mix = float(h_mix_values[stability_class])

    z_char = 0.1 * h_mix

    if L < 0.0:
        w_star = u_star * (h_mix / abs(L))**(1.0 / 3.0)
        D_horizontal = 0.1 * w_star * h_mix
    else:
        D_horizontal = 0.4 * u_star * z_char

    if L < 0.0:
        D_vertical = 0.1 * u_star * h_mix * (z_char / abs(L))**(1.0 / 3.0)
    elif abs(L) > 1e5:
        D_vertical = 0.4 * u_star * z_char
    else:
        D_vertical = 0.4 * u_star * z_char * (1.0 + 5.0 * z_char / L)**(-1.0)

    D_horizontal = float(np.clip(D_horizontal, 1.0, 100.0))
    D_vertical = float(np.clip(D_vertical, 0.1, 50.0))
    return D_horizontal, D_vertical


def get_pasquill_class(wind_speed: float,
                      solar_radiation: float,
                      cloud_cover: float,
                      is_daytime: bool) -> str:
    """Turner (EPA AP-26) Pasquill-Gifford-Turner stability classification."""
    wind_speed = float(wind_speed)
    solar_radiation = float(solar_radiation)
    cloud_cover = float(cloud_cover)

    if is_daytime:
        if solar_radiation > 600.0:
            insolation = 'strong'
        elif solar_radiation > 300.0:
            insolation = 'moderate'
        else:
            insolation = 'slight'

        if insolation == 'strong':
            if wind_speed < 2.0:
                return 'A'
            if wind_speed < 3.0:
                return normalize_stability_class('A-B')
            if wind_speed < 5.0:
                return 'B'
            if wind_speed < 6.0:
                return 'C'
            return 'D'

        if insolation == 'moderate':
            if wind_speed < 2.0:
                return normalize_stability_class('A-B')
            if wind_speed < 3.0:
                return 'B'
            if wind_speed < 5.0:
                return normalize_stability_class('B-C')
            if wind_speed < 6.0:
                return 'C'
            return 'D'

        if wind_speed < 5.0:
            return 'C'
        return 'D'

    if cloud_cover >= 50.0:
        return 'D'
    if wind_speed < 2.5:
        return 'F'
    if wind_speed < 3.5:
        return 'E'
    return 'D'


def normalize_stability_class(class_string: str) -> str:
    class_string = str(class_string).strip().upper()
    if '-' in class_string:
        _, c2 = class_string.split('-', 1)
        return c2.strip()[0]
    return class_string[0]


def sample_meteorological_conditions(wind_speed: float, rng):
    """
    Sample realistic meteorological conditions
    Returns: (solar_radiation, cloud_cover, hour, is_daytime)
    """
    # Sample hour
    hour = rng.uniform(0, 24)
    is_daytime = 6 <= hour <= 18
    
    if is_daytime:
        # Sample stability class (weighted by frequency)
        stability_choices = ['D', 'C', 'B', 'A']
        stability_weights = [0.50, 0.30, 0.15, 0.05]
        stability = rng.choice(stability_choices, p=stability_weights)
        
        # Time factor for solar radiation
        time_factor = np.sin(np.pi * (hour - 6) / 12)
        
        # Solar radiation and cloud cover based on stability
        if stability == 'A':
            solar = rng.uniform(600, 1000) * time_factor
            cloud = rng.uniform(0, 20)
        elif stability == 'B':
            solar = rng.uniform(450, 650) * time_factor
            cloud = rng.uniform(0, 30)
        elif stability == 'C':
            solar = rng.uniform(250, 450) * time_factor
            cloud = rng.uniform(20, 50)
        else:  # D
            solar = rng.uniform(0, 300) * time_factor
            cloud = rng.uniform(40, 80)
    else:
        # Nighttime
        stability_choices = ['D', 'E', 'F']
        stability_weights = [0.50, 0.35, 0.15]
        stability = rng.choice(stability_choices, p=stability_weights)
        
        solar = 0.0
        if stability == 'F':
            cloud = rng.uniform(0, 30)
        elif stability == 'E':
            cloud = rng.uniform(20, 60)
        else:
            cloud = rng.uniform(50, 100)
    
    return solar, cloud, hour, is_daytime


def get_vertical_velocity(stability_class: str, wind_speed: float) -> float:
    """Vertical velocity from atmospheric turbulence and buoyancy"""
    # Much smaller values for 5m domain - keep material in sampling plane
    if stability_class in ['A', 'B']:  # Unstable
        return 0.001  # Very gentle updrafts (1mm/s)
    elif stability_class in ['C', 'D']:  # Neutral
        return 0.0005  # Minimal mechanical turbulence (0.5mm/s)
    else:  # E, F - Stable
        return 0.0001  # Almost no vertical motion (0.1mm/s)


def check_steady_state_mass(history, Q_total, dt, tolerance=0.001, window=10):
    """Check if total MASS has plateaued (better steady-state detection).
    
    Steady state is reached when mass growth rate ≈ emission rate,
    meaning mass entering (Q_total) ≈ mass leaving (outflow).
    
    Args:
        history: List of metric dictionaries with 'total_mass' and 'time' keys
        Q_total: Total emission rate (kg/s)
        dt: Time step size (s)
        tolerance: Relative tolerance for mass balance (default: 0.001 = 0.1%)
        window: Number of recent snapshots to analyze (default: 10)
    
    Returns:
        True if steady state detected, False otherwise
    """
    if len(history) < window:
        return False
    
    recent_mass = [h['total_mass'] for h in history[-window:]]
    recent_times = [h['time'] for h in history[-window:]]
    
    # Check if all values are finite
    if not all(np.isfinite(m) for m in recent_mass):
        return False
    
    # Calculate mass growth rate (kg/s)
    # Use actual time differences for accuracy
    total_time_span = recent_times[-1] - recent_times[0]
    if total_time_span <= 0:
        return False
    
    mass_growth_rate = (recent_mass[-1] - recent_mass[0]) / total_time_span
    
    # Steady state when growth rate ≈ emission rate (mass leaving ≈ mass entering)
    # For a steady state, the mass growth should be near zero (or very small)
    # since input = output. However, if we're still building up, growth ≈ Q_total
    # We check if the NET growth (after accounting for emission) is small
    if Q_total > 0:
        # Relative error: how close is growth rate to emission rate?
        relative_error = abs(mass_growth_rate - Q_total) / Q_total
        # Steady state: growth rate should be very small compared to emission
        # (most mass is leaving, little is accumulating)
        return relative_error < tolerance or abs(mass_growth_rate) < tolerance * Q_total
    else:
        # No emission: check if mass is constant
        mass_change = abs(recent_mass[-1] - recent_mass[0])
        mean_mass = np.mean(recent_mass)
        if mean_mass == 0.0:
            return True
        return (mass_change / mean_mass) < tolerance
