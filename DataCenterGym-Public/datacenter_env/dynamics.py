
"""
Thermal dynamics and cooling controllers for datacenter simulation.
"""
import numpy as np


class CoolingController:
    """Base class for cooling control strategies."""

    def __init__(self, cooling_max, T_target):
        """
        Args:
            cooling_max: Maximum cooling power (W)
            T_target: Target internal temperature (°C)
        """
        self.cooling_max = cooling_max
        self.T_target = T_target

    def compute_cooling_power(self, theta_current, dt):
        """
        Compute cooling power based on current temperature.

        Args:
            theta_current: Current datacenter temperature (°C)
            dt: Time step (seconds)

        Returns:
            cooling_power: Cooling power in Watts (non-negative)
        """
        raise NotImplementedError


class ProportionalController(CoolingController):
    """
    Bidirectional proportional HVAC controller.

    HVAC power is proportional to temperature error:
        hvac_power = K_p × (θ_current - T_target)

    Positive power = cooling (removes heat)
    Negative power = heating (adds heat)

    This models realistic datacenter HVAC systems that can both
    heat and cool to maintain target temperature.
    """

    def __init__(self, cooling_max, T_target, K_p=1000.0, heating_max=None):
        """
        Args:
            cooling_max: Maximum cooling power (W)
            T_target: Target internal temperature (°C)
            K_p: Proportional gain (W/°C). Higher = more aggressive control.
            heating_max: Maximum heating power (W). If None, uses cooling_max.
        """
        super().__init__(cooling_max, T_target)
        self.K_p = K_p
        self.heating_max = heating_max if heating_max is not None else cooling_max

    def compute_cooling_power(self, theta_current, dt):
        """
        Compute bidirectional HVAC power.

        Returns:
            hvac_power: Positive = cooling, Negative = heating
        """
        # Full temperature error (not clamped to 0)
        error = theta_current - self.T_target
        hvac_power = self.K_p * error

        # Clamp to [-heating_max, +cooling_max]
        hvac_power = np.clip(hvac_power, -self.heating_max, self.cooling_max)

        return hvac_power


class PIDController(CoolingController):
    """
    PID (Proportional-Integral-Derivative) cooling controller.

    Cooling power components:
        - Proportional: Responds to current error
        - Integral: Eliminates steady-state error
        - Derivative: Reduces overshoot and oscillation

    Formula:
        cooling_power = K_p × e(t) + K_i × ∫e(t)dt + K_d × de(t)/dt

    where e(t) = max(0, θ_current - T_target)
    """

    def __init__(self, cooling_max, T_target, K_p=1000.0, K_i=50.0, K_d=500.0):
        """
        Args:
            cooling_max: Maximum cooling power (W)
            T_target: Target internal temperature (°C)
            K_p: Proportional gain (W/°C)
            K_i: Integral gain (W/(°C·s))
            K_d: Derivative gain (W·s/°C)
        """
        super().__init__(cooling_max, T_target)
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d

        # State variables
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_cooling_power(self, theta_current, dt):
        """Compute PID cooling power."""
        # Only cool when above target (one-sided control)
        error = max(0, theta_current - self.T_target)

        # Proportional term
        P = self.K_p * error

        # Integral term (accumulate error over time)
        # Only accumulate if we're above target AND not at max cooling
        # Reset integral if temperature is below target (prevents overcooling)
        if theta_current < self.T_target:
            self.integral = 0.0  # Reset integral when below target
        else:
            self.integral += error * dt
            # Anti-windup: clamp integral to prevent excessive buildup
            max_integral = self.cooling_max / (self.K_i + 1e-6)
            self.integral = np.clip(self.integral, 0.0, max_integral)

        I = self.K_i * self.integral

        # Derivative term (rate of change of error)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        D = self.K_d * derivative

        # Update state
        self.prev_error = error

        # Total cooling power
        cooling_power = P + I + D

        # Clamp to [0, cooling_max]
        cooling_power = np.clip(cooling_power, 0.0, self.cooling_max)

        return cooling_power

    def reset(self):
        """Reset controller state (call when episode resets)."""
        self.integral = 0.0
        self.prev_error = 0.0


def thermal_dynamics_step(theta, T_amb, heat_generation, cooling_power, R, C, dt):
    """
    Update datacenter temperature using thermal RC model.

    Thermal equation:
        C × dθ/dt = Q_gen - (θ - T_amb)/R - Q_hvac

    Args:
        theta: Current temperature (°C)
        T_amb: Ambient temperature (°C)
        heat_generation: Heat generated by IT equipment (W)
        cooling_power: HVAC power (W). Positive = cooling, Negative = heating
        R: Thermal resistance (°C/W)
        C: Thermal capacitance (J/°C)
        dt: Time step (seconds)

    Returns:
        theta_new: Updated temperature (°C)
    """
    # Heat dissipation to ambient through building envelope
    heat_dissipation = (theta - T_amb) / R

    # Net heat flow
    net_heat = heat_generation - heat_dissipation - cooling_power

    # Temperature change: dθ = (1/C) × Q × dt
    dtheta = (net_heat / C) * dt

    theta_new = theta + dtheta

    return theta_new


def create_cooling_controller(controller_type, cooling_max, T_target, params=None):
    """
    Factory function to create cooling controller.

    Args:
        controller_type: "proportional" or "pid"
        cooling_max: Maximum cooling power (W)
        T_target: Target temperature (°C)
        params: Dict of controller-specific parameters

    Returns:
        CoolingController instance
    """
    params = params or {}

    if controller_type.lower() == "proportional":
        K_p = params.get("K_p", 1000.0)
        return ProportionalController(cooling_max, T_target, K_p)

    elif controller_type.lower() == "pid":
        K_p = params.get("K_p", 1000.0)
        K_i = params.get("K_i", 50.0)
        K_d = params.get("K_d", 500.0)
        return PIDController(cooling_max, T_target, K_p, K_i, K_d)

    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
