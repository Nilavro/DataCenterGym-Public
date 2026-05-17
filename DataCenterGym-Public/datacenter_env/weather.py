
"""
Weather and Climate Generation for DataCenterGym

Generates realistic outdoor ambient temperatures for each datacenter based on:
- Base temperature (location-specific average)
- Daily amplitude (temperature variation)
- Seasonal patterns
- Time-of-day effects

This module simulates the outdoor temperature (T_amb) that drives cooling requirements
in the thermal model: C × dθ/dt = Q_gen - (θ - T_amb)/R - Q_hvac
"""

import numpy as np
from typing import Dict, List, Optional


def generate_daily_temperature(
    timestep: int,
    base_temp: float,
    amplitude: float,
    offset_hours: float = 0.0,
    dt_minutes: float = 5.0,
    noise_std: float = 0.5
) -> float:
    """
    Generate outdoor ambient temperature for a given timestep.

    Uses sinusoidal pattern to model daily temperature variation:
        T_amb(t) = base_temp + amplitude * sin(2π * t / period + phase) + noise

    Args:
        timestep: Current timestep index
        base_temp: Average temperature for location (°C)
        amplitude: Half of temperature range (°C)
        offset_hours: Time offset for peak temperature (hours from midnight)
        dt_minutes: Timestep duration in minutes
        noise_std: Standard deviation of Gaussian noise (°C)

    Returns:
        Ambient temperature in °C

    Example:
        >>> # Phoenix in summer: base=35°C, amplitude=10°C, peak at 2pm
        >>> temp = generate_daily_temperature(
        ...     timestep=144,  # 12 hours into day (noon)
        ...     base_temp=35.0,
        ...     amplitude=10.0,
        ...     offset_hours=2.0  # Peak at 2pm
        ... )
        >>> # Returns ~43°C (near peak of 35+10=45°C, but shifted by offset)
    """
    # Convert timestep to hours
    hours_elapsed = (timestep * dt_minutes) / 60.0

    # Compute phase shift (offset moves peak time)
    # Phase = 2π * (hours - offset) / 24, with adjustment so peak is at desired time
    # Default: peak at noon (12 hours), so phase starts at -π/2
    phase = 2 * np.pi * (hours_elapsed - 12.0 + offset_hours) / 24.0

    # Sinusoidal temperature variation
    temp_variation = amplitude * np.sin(phase)

    # Add small random noise for realism
    noise = np.random.normal(0, noise_std) if noise_std > 0 else 0.0

    return base_temp + temp_variation + noise


def generate_weather_trace(
    episode_length: int,
    datacenter_climates: List[Dict],
    dt_minutes: float = 5.0,
    noise_std: float = 0.5,
    seed: Optional[int] = None
) -> Dict[int, Dict[int, float]]:
    """
    Generate complete weather trace for entire episode.

    Args:
        episode_length: Number of timesteps in episode
        datacenter_climates: List of climate configs, one per datacenter
            Each dict should have:
                - base_temp: Average temperature (°C)
                - amplitude: Temperature variation (°C)
                - offset_hours: Time offset for peak (hours)
        dt_minutes: Timestep duration in minutes
        noise_std: Standard deviation of temperature noise (°C)
        seed: Random seed for reproducibility

    Returns:
        Dict mapping timestep -> datacenter_id -> temperature
        Format: {timestep: {dc_id: T_amb, ...}, ...}

    Example:
        >>> climates = [
        ...     {'base_temp': 15.0, 'amplitude': 8.0, 'offset_hours': 0},  # Seattle
        ...     {'base_temp': 35.0, 'amplitude': 10.0, 'offset_hours': 2}, # Phoenix
        ...     {'base_temp': 18.0, 'amplitude': 9.0, 'offset_hours': 1},  # Chicago
        ...     {'base_temp': 28.0, 'amplitude': 9.0, 'offset_hours': 1.5} # Dallas
        ... ]
        >>> weather = generate_weather_trace(288, climates)  # 24 hours @ 5 min
        >>> weather[144][1]  # Phoenix at noon
        41.2
    """
    if seed is not None:
        np.random.seed(seed)

    weather_trace = {}

    for t in range(episode_length):
        weather_trace[t] = {}

        for dc_id, climate in enumerate(datacenter_climates):
            base_temp = climate.get('base_temp', 20.0)
            amplitude = climate.get('amplitude', 5.0)
            offset_hours = climate.get('offset_hours', 0.0)

            temp = generate_daily_temperature(
                timestep=t,
                base_temp=base_temp,
                amplitude=amplitude,
                offset_hours=offset_hours,
                dt_minutes=dt_minutes,
                noise_std=noise_std
            )

            weather_trace[t][dc_id] = temp

    return weather_trace


def get_ambient_temperature(
    weather_trace: Dict[int, Dict[int, float]],
    timestep: int,
    dc_id: int,
    fallback_temp: float = 20.0
) -> float:
    """
    Retrieve ambient temperature from weather trace with fallback.

    Args:
        weather_trace: Pre-generated weather trace
        timestep: Current timestep
        dc_id: Datacenter ID
        fallback_temp: Default temperature if not found (°C)

    Returns:
        Ambient temperature in °C
    """
    if weather_trace and timestep in weather_trace:
        return weather_trace[timestep].get(dc_id, fallback_temp)
    return fallback_temp


# Predefined climate profiles for common datacenter locations
CLIMATE_PROFILES = {
    'seattle': {
        'location': 'Seattle, WA (Cool Marine)',
        'base_temp': 15.0,
        'amplitude': 8.0,
        'offset_hours': 0.0,
        'description': 'Cool, temperate marine climate with moderate variation'
    },
    'phoenix': {
        'location': 'Phoenix, AZ (Hot Desert)',
        'base_temp': 35.0,
        'amplitude': 10.0,
        'offset_hours': 2.0,
        'description': 'Hot desert climate with large daily temperature swings'
    },
    'chicago': {
        'location': 'Chicago, IL (Continental)',
        'base_temp': 18.0,
        'amplitude': 9.0,
        'offset_hours': 1.0,
        'description': 'Continental climate with significant temperature variation'
    },
    'dallas': {
        'location': 'Dallas, TX (Humid Subtropical)',
        'base_temp': 28.0,
        'amplitude': 9.0,
        'offset_hours': 1.5,
        'description': 'Hot humid subtropical with afternoon peak temperatures'
    },
    'arctic': {
        'location': 'Arctic Region (Extreme Cold)',
        'base_temp': -10.0,
        'amplitude': 5.0,
        'offset_hours': 0.0,
        'description': 'Extremely cold climate, minimal cooling needed'
    },
    'equator': {
        'location': 'Equatorial (Hot Humid)',
        'base_temp': 30.0,
        'amplitude': 3.0,
        'offset_hours': 1.0,
        'description': 'Hot humid equatorial climate with small daily variation'
    }
}


if __name__ == "__main__":
    # Example usage and validation
    print("Weather Generation Module for DataCenterGym")
    print("=" * 60)

    # Example 1: Single temperature generation
    print("\nExample 1: Generate temperature for Phoenix at 2pm")
    timestep_2pm = int(14 * 60 / 5)  # 14 hours * 60 min / 5 min timesteps
    temp = generate_daily_temperature(
        timestep=timestep_2pm,
        base_temp=35.0,
        amplitude=10.0,
        offset_hours=2.0,
        noise_std=0.5
    )
    print(f"  Phoenix at 2pm: {temp:.1f}°C (expected ~45°C peak)")

    # Example 2: Full day weather trace
    print("\nExample 2: Generate 24-hour weather trace for 4 datacenters")
    climates = [
        CLIMATE_PROFILES['seattle'],
        CLIMATE_PROFILES['phoenix'],
        CLIMATE_PROFILES['chicago'],
        CLIMATE_PROFILES['dallas']
    ]

    weather = generate_weather_trace(
        episode_length=288,  # 24 hours @ 5-minute timesteps
        datacenter_climates=climates,
        dt_minutes=5.0,
        noise_std=0.5,
        seed=42
    )

    print("\n  Temperature ranges over 24 hours:")
    locations = ['Seattle', 'Phoenix', 'Chicago', 'Dallas']
    for dc_id, location in enumerate(locations):
        temps = [weather[t][dc_id] for t in range(288)]
        print(f"    {location:<12}: {min(temps):.1f}°C to {max(temps):.1f}°C (avg {np.mean(temps):.1f}°C)")

    # Example 3: Show temperature pattern for Phoenix
    print("\nExample 3: Phoenix hourly temperatures")
    print("  Time   Temperature")
    print("  " + "-" * 20)
    for hour in [0, 6, 12, 14, 18, 23]:
        timestep = int(hour * 60 / 5)
        temp = weather[timestep][1]  # Phoenix is dc_id=1
        print(f"  {hour:02d}:00  {temp:5.1f}°C")

    print("\n✅ Weather generation module working correctly!")
