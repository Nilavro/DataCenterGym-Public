import gymnasium as gym
import numpy as np
from gymnasium import spaces
from math import sin, pi
from .dynamics import create_cooling_controller, thermal_dynamics_step

class DataCenterEnv(gym.Env):
    def __init__(self, cluster_map, job_trace, usage_trace, machine_meta, config):
        super().__init__()

        self.cluster_map = cluster_map
        self.job_trace = job_trace
        self.usage_trace = usage_trace
        self.machine_meta = machine_meta
        self.config = config

        self.T = config.get("episode_length", 288)
        self.dt = config.get("time_step_minutes", 5)
        self.M = config.get("num_clusters", len(set(cluster_map.values())))
        self.D = config.get("num_datacenters", 1)
        self.weights = config.get("lambda_weights", {})

        # Price signal configuration
        self.pricing_config = config.get("pricing", {})
        self.pricing_enabled = self.pricing_config.get("enabled", False)
        self.pricing_model = self.pricing_config.get("model", "time_of_use")  # "time_of_use", "sinusoid"

        # Per-datacenter pricing or global
        self.per_datacenter_pricing = self.pricing_config.get("per_datacenter", False)

        # cluster to datacenter mapping (as list and dict)
        self.cluster_to_dc_list = config.get("cluster_to_dc", [0] * self.M)
        self.cluster_to_dc = {i: dc_id for i, dc_id in enumerate(self.cluster_to_dc_list)}

        # Initialize datacenters with cooling controllers
        dc_ids = set(self.cluster_to_dc_list)
        self.datacenters = {}
        self.cooling_controllers = {}

        for dc_id in dc_ids:
            dc_params = config['datacenter_params'][dc_id]

            # Create cooling controller
            controller_type = dc_params.get('controller_type', 'proportional')
            cooling_max = dc_params.get('cooling_max', 10000)
            T_target = dc_params.get('T_target', 22)
            controller_params = dc_params.get('controller_params', {})

            self.cooling_controllers[dc_id] = create_cooling_controller(
                controller_type, cooling_max, T_target, controller_params
            )

            # Thermal throttling parameters
            throttling_config = dc_params.get('thermal_throttling', {})
            throttling_enabled = throttling_config.get('enabled', False)
            theta_soft = throttling_config.get('theta_soft', 70.0)
            theta_max = dc_params.get('theta_max', 80.0)
            g_min = throttling_config.get('g_min', 0.5)

            self.datacenters[dc_id] = {
                "theta": dc_params.get('theta_init', 30.0),
                "T_amb": 25.0,
                "u": 0.0,
                "cooling_power": 0.0,  # Actual cooling power applied (W)
                "T_target": T_target,   # Target temperature setpoint
                "throttling_enabled": throttling_enabled,
                "theta_soft": theta_soft,
                "theta_max": theta_max,
                "g_min": g_min
            }

        high = np.array([100.0, 500.0, 1000.0, 500.0, 60.0] * self.M + [1.0] * self.D)
        low = np.array([0.0] * len(high))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_jobs_per_step = 100
        self.action_space = spaces.Dict({
            "job_assignment": spaces.MultiDiscrete([self.M + 1] * self.max_jobs_per_step),
            "T_target": spaces.Box(low=18.0, high=28.0, shape=(self.D,), dtype=np.float32)  # Target temp setpoints
        })

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.clusters = [self._init_cluster_state(i) for i in range(self.M)]

        # Reset datacenters and controllers
        for dc_id, dc in self.datacenters.items():
            dc_params = self.config['datacenter_params'][dc_id]
            throttling_config = dc_params.get('thermal_throttling', {})
            dc.update({
                "theta": dc_params.get('theta_init', 30.0),
                "T_amb": 25.0,
                "u": 0.0,
                "cooling_power": 0.0,
                "T_target": dc_params.get('T_target', 22),
                "throttling_enabled": throttling_config.get('enabled', False),
                "theta_soft": throttling_config.get('theta_soft', 70.0),
                "theta_max": dc_params.get('theta_max', 80.0),
                "g_min": throttling_config.get('g_min', 0.5)
            })
            # Reset PID controller state if applicable
            if hasattr(self.cooling_controllers[dc_id], 'reset'):
                self.cooling_controllers[dc_id].reset()

        # Reset cost tracking
        self.total_energy_kwh = 0.0  # Total energy consumed (kWh)
        self.total_cost_usd = 0.0  # Total cost ($)
        self.step_energy_history = []  # Energy per step (kWh)
        self.step_cost_history = []  # Cost per step ($)
        self.step_price_history = []  # Price per step ($/kWh)

        # Reset job completion tracking
        self.n_completed = 0  # Total jobs completed
        self.n_arrived = 0  # Total jobs arrived

        self.current_jobs = self.job_trace.get(self.t, [])
        return self._get_obs(), {}

    def step(self, action):
        job_action = action["job_assignment"]
        T_target_action = action["T_target"]

        assert len(job_action) == len(self.current_jobs)
        assert len(T_target_action) == self.D

        # Track job arrivals
        self.n_arrived += len(self.current_jobs)

        # Update target temperatures and controller setpoints
        for dc_id, T_target in enumerate(T_target_action):
            self.datacenters[dc_id]["T_target"] = T_target
            self.cooling_controllers[dc_id].T_target = T_target

        for job, a in zip(self.current_jobs, job_action):
            if a < self.M:
                # Check affinity constraint if enabled
                if self.config.get("enforce_affinity", False):
                    cluster_type = self.config["cluster_params"][a].get("type", "cpu")
                    job_type = job.get("tau", "cpu")

                    if cluster_type != job_type:
                        # Invalid assignment - skip this job (stays in queue)
                        continue

                self._assign_job_to_cluster(job, a)

        for i in range(self.M):
            self._evolve_cluster(i)

        for dc_id in self.datacenters:
            self._evolve_datacenter(dc_id)

        # Compute energy and cost for this step (after all power is computed)
        step_energy, step_cost, step_price = self._compute_step_energy_and_cost()
        self.total_energy_kwh += step_energy
        self.total_cost_usd += step_cost
        self.step_energy_history.append(step_energy)
        self.step_cost_history.append(step_cost)
        self.step_price_history.append(step_price)

        self.t += 1
        terminated = self.t >= self.T
        truncated = False
        self.current_jobs = self.job_trace.get(self.t, [])

        obs = self._get_obs()
        reward = self._compute_reward()
        info = {
            'step_energy_kwh': step_energy,
            'step_cost_usd': step_cost,
            'step_price_per_kwh': step_price,
            'total_energy_kwh': self.total_energy_kwh,
            'total_cost_usd': self.total_cost_usd,
            'n_completed': self.n_completed,
            'n_arrived': self.n_arrived
        }
        return obs, reward, terminated, truncated, info

    def _init_cluster_state(self, i):
        return {
            "p": 100.0,
            "c": 200.0,
            "q": 0.0,
            "u": 0.0,
            "jobs": []
        }

    def _assign_job_to_cluster(self, job, cluster_id):
        self.clusters[cluster_id]["jobs"].append(job)
        self.clusters[cluster_id]["q"] += 1

    def _evolve_cluster(self, i):
        cluster = self.clusters[i]
        params = self.config["cluster_params"][i]

        # Get datacenter ID for this cluster
        dc_id = self.cluster_to_dc[i]

        # Apply thermal throttling to compute capacity
        g = self._get_throttling_factor(dc_id)
        c_effective = params['c_max'] * g  # Effective capacity after throttling

        # FIXED: Only count resources up to capacity for heat/power generation
        # Jobs exceeding capacity are queued and don't generate heat
        u_total = sum([j['r'] for j in cluster['jobs']])  # Total resource demand
        cluster['c'] = c_effective  # Store effective capacity
        cluster['c_max'] = params['c_max']  # Store nominal capacity for reference
        cluster['g'] = g  # Store throttling factor for metrics
        u_running = min(u_total, cluster['c'])  # Cap at effective capacity

        cluster['u'] = u_running  # Used for heat generation (realistic)
        cluster['u_total'] = u_total  # Track total demand for metrics
        cluster['p'] -= params['phi'] * u_running  # Only running jobs consume power

        # CAPACITY-CONSTRAINED PROCESSING WITH BACKFILLING:
        # Jobs that fit within cluster capacity make progress.
        # If a job doesn't fit, it stays queued but doesn't block smaller jobs behind it.
        # This implements "backfilling" - a common scheduler optimization.
        processing_jobs = []
        waiting_jobs = []
        n_completed_this_cluster = 0
        capacity_used = 0.0

        # First pass: identify which jobs can be processed
        for job in cluster['jobs']:
            job_resources = job['r']

            # Check if this job can be processed (fits within remaining capacity)
            if capacity_used + job_resources <= c_effective:
                # Job will be processed this timestep
                capacity_used += job_resources
                processing_jobs.append(job)
            else:
                # Job doesn't fit now - add to waiting list
                waiting_jobs.append(job)

        # Second pass: process the jobs
        remaining_jobs = []
        for job in processing_jobs:
            job['remaining'] = job.get('remaining', job['d']) - 1

            if job['remaining'] > 0:
                remaining_jobs.append(job)
            else:
                n_completed_this_cluster += 1

        # Add waiting jobs back to queue (they don't make progress)
        remaining_jobs.extend(waiting_jobs)

        # Track completed jobs
        self.n_completed += n_completed_this_cluster

        cluster['jobs'] = remaining_jobs
        cluster['q'] = len(remaining_jobs)

    def _evolve_datacenter(self, dc_id):
        dc = self.datacenters[dc_id]
        cluster_indices = [i for i, d in self.cluster_to_dc.items() if d == dc_id]
        total_u = sum(self.clusters[i]['u'] for i in cluster_indices)
        dc['u'] = total_u

        params = self.config['datacenter_params'][dc_id]
        R, C = params['R'], params['C']

        # Calculate heat generation from each cluster with its own alpha
        heat_generation = 0.0
        for cluster_idx in cluster_indices:
            alpha_i = self.config["cluster_params"][cluster_idx]["alpha"]
            heat_generation += alpha_i * self.clusters[cluster_idx]['u']

        theta = dc['theta']
        T_amb = self.config.get('weather_trace', {}).get(self.t, {}).get(dc_id, dc['T_amb'])
        dc['T_amb'] = T_amb

        # Use cooling controller to compute cooling power
        dt_seconds = self.dt * 60  # Convert minutes to seconds
        controller = self.cooling_controllers[dc_id]
        cooling_power = controller.compute_cooling_power(theta, dt_seconds)
        dc['cooling_power'] = cooling_power

        # Update temperature using thermal dynamics
        dc['theta'] = thermal_dynamics_step(
            theta, T_amb, heat_generation, cooling_power, R, C, dt_seconds
        )

    def _get_obs(self):
        obs = []
        for i, c in enumerate(self.clusters):
            dc_id = self.cluster_to_dc[i]
            obs.extend([
                self.datacenters[dc_id]['theta'],
                c['p'], c['c'], c['q'],
                self.datacenters[dc_id]['T_amb']
            ])
        for dc_id in range(self.D):
            obs.append(self.datacenters[dc_id]['T_target'])
        return np.array(obs, dtype=np.float32)

    def _get_throttling_factor(self, dc_id):
        """
        Compute thermal throttling factor g(theta) for datacenter dc_id.

        Returns a factor in [g_min, 1] that reduces effective compute capacity
        as temperature exceeds theta_soft.

        Args:
            dc_id: Datacenter ID

        Returns:
            Throttling factor g(theta) in [g_min, 1]
        """
        dc = self.datacenters[dc_id]

        # If throttling is disabled, return full capacity
        if not dc['throttling_enabled']:
            return 1.0

        theta = dc['theta']
        theta_soft = dc['theta_soft']
        theta_max = dc['theta_max']
        g_min = dc['g_min']

        # Piecewise linear throttling function
        if theta <= theta_soft:
            return 1.0
        elif theta >= theta_max:
            return g_min
        else:
            # Linear interpolation between 1.0 and g_min
            return 1.0 - (1.0 - g_min) * (theta - theta_soft) / (theta_max - theta_soft)

    def _compute_reward(self):
        w = self.weights
        reward = 0.0
        for dc_id, dc in self.datacenters.items():
            overheat = max(0, dc['theta'] - self.config['datacenter_params'][dc_id]['theta_max'])
            reward -= w.get('theta', 1.0) * overheat
            # Penalize cooling power (normalized by cooling_max)
            cooling_max = self.config['datacenter_params'][dc_id].get('cooling_max', 30000)
            cooling_fraction = dc['cooling_power'] / cooling_max
            reward -= w.get('cooling', 0.1) * cooling_fraction
        for c in self.clusters:
            reward -= w.get('q', 0.5) * c['q']
            reward -= w.get('energy', 0.2) * (c['p'] ** 2)
        for job in self.current_jobs:
            reward += job['v']
        return reward

    def _get_price_signal(self, t, dc_id=None):
        """
        Get electricity price at time t for datacenter dc_id.

        Args:
            t: Current time step
            dc_id: Datacenter ID (None for global pricing)

        Returns:
            Price in $/kWh
        """
        if not self.pricing_enabled:
            return 0.0

        # Get pricing parameters
        if self.per_datacenter_pricing and dc_id is not None:
            price_params = self.pricing_config.get('datacenters', {}).get(dc_id, {})
        else:
            price_params = self.pricing_config

        if self.pricing_model == "time_of_use":
            return self._time_of_use_price(t, price_params)
        elif self.pricing_model == "sinusoid":
            return self._sinusoid_price(t, price_params)
        else:
            # Default to constant price
            return price_params.get('base_price', 0.10)

    def _time_of_use_price(self, t, params):
        """
        Time-of-use pricing: two-level day/night prices.

        Typical structure:
        - Peak hours (day): 8am-8pm
        - Off-peak hours (night): 8pm-8am

        Args:
            t: Time step
            params: Pricing parameters dict

        Returns:
            Price in $/kWh
        """
        # Convert time step to hour of day
        minutes_per_step = self.dt
        total_minutes = t * minutes_per_step
        hour_of_day = (total_minutes // 60) % 24

        peak_price = params.get('peak_price', 0.15)  # $/kWh during peak
        offpeak_price = params.get('offpeak_price', 0.08)  # $/kWh during off-peak
        peak_start = params.get('peak_start_hour', 8)  # 8am
        peak_end = params.get('peak_end_hour', 20)  # 8pm

        if peak_start <= hour_of_day < peak_end:
            return peak_price
        else:
            return offpeak_price

    def _sinusoid_price(self, t, params):
        """
        Sinusoidal pricing: smooth periodic variation.

        Price follows: base + amplitude * sin(2π * t / period + phase)

        Args:
            t: Time step
            params: Pricing parameters dict

        Returns:
            Price in $/kWh
        """
        base_price = params.get('base_price', 0.12)  # $/kWh average
        amplitude = params.get('amplitude', 0.05)  # $/kWh variation
        period_steps = params.get('period_steps', 288)  # Default: 24 hours at 5-min steps
        phase_offset = params.get('phase_offset', 0.0)  # Radians

        # Sinusoidal variation (peak at noon, low at midnight)
        phase = (2 * pi * t / period_steps) + phase_offset
        price = base_price + amplitude * sin(phase)

        # Ensure non-negative price
        return max(0.0, price)

    def _compute_step_energy_and_cost(self):
        """
        Compute energy consumption and cost for the current step.

        Returns:
            (energy_kwh, cost_usd, avg_price) tuple
        """
        if not self.pricing_enabled:
            return 0.0, 0.0, 0.0

        dt_hours = self.dt / 60.0  # Convert minutes to hours

        total_energy_kwh = 0.0
        total_cost_usd = 0.0

        for dc_id in self.datacenters:
            dc = self.datacenters[dc_id]
            cluster_indices = [i for i, d in self.cluster_to_dc.items() if d == dc_id]

            # Compute power for this datacenter
            compute_power = 0.0
            for cluster_idx in cluster_indices:
                alpha_i = self.config["cluster_params"][cluster_idx]["alpha"]
                compute_power += alpha_i * self.clusters[cluster_idx]['u']

            cooling_power = dc['cooling_power']
            total_power = compute_power + cooling_power  # Watts

            # Energy consumed this step
            energy_kwh = (total_power / 1000.0) * dt_hours  # Convert W to kW, multiply by hours

            # Get price for this datacenter
            price = self._get_price_signal(self.t, dc_id if self.per_datacenter_pricing else None)

            # Cost for this datacenter
            cost_usd = energy_kwh * price

            total_energy_kwh += energy_kwh
            total_cost_usd += cost_usd

        # Compute average price across datacenters
        avg_price = total_cost_usd / total_energy_kwh if total_energy_kwh > 0 else 0.0

        return total_energy_kwh, total_cost_usd, avg_price

    def render(self):
        for i, c in enumerate(self.clusters):
            dc_id = self.cluster_to_dc[i]
            dc = self.datacenters[dc_id]
            cooling_kw = dc['cooling_power'] / 1000  # Convert to kW
            print(f"Cluster {i} | DC {dc_id}: θ={dc['theta']:.1f}°C (target={dc['T_target']:.1f}°C), p={c['p']:.1f}, q={c['q']} jobs | Cooling={cooling_kw:.1f}kW")
