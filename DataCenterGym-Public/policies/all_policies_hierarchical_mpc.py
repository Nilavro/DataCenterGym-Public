"""
Hierarchical Two-Stage MPC Scheduler

Implements a true hierarchical two-stage MPC where both stages use MPC optimization
to reduce variable count while maintaining optimality.

Architecture:
- Stage 1: DC-level MPC optimizes allocation ratios + temperature setpoints (16 variables)
- Stage 2: Per-DC MPC refines job-cluster assignments constrained by Stage 1

Expected benefits:
- 8-67x speedup vs monolithic MPC
- Handles 200+ jobs (vs 50 job limit)
- Both stages MPC-optimized (better than DC-Ratio's greedy Stage 2)
- Optimizes temperature setpoints (better than SCMPCScheduler's fixed temps)
"""

import numpy as np
import casadi as ca
import time
from typing import Dict, List, Tuple, Optional, Any
from policies.all_policies import SchedulingPolicy


class HierarchicalMPCStage1:
    """
    Stage 1: DC-Level MPC Optimization.

    Optimizes datacenter-level allocation ratios and temperature setpoints.
    Decision variables: p_cpu[4], p_gpu[4], T_target[4], xi[4] = 16 total
    """

    def __init__(self, env, N: int = 6, weights: Optional[Dict[str, float]] = None, verbose: bool = False):
        """
        Initialize Stage 1 DC-level MPC.

        Args:
            env: DataCenterEnv instance
            N: MPC horizon length (6 timesteps = 30 min)
            weights: Dict with keys: wQ, wT, wE, wL, wS
            verbose: Print debug information
        """
        self.env = env
        self.N = N
        self.weights = weights or self._default_weights()
        self.verbose = verbose

        # Environment dimensions
        self.D = env.D  # Number of datacenters
        self.M = env.M  # Number of clusters

        # Load DC-level parameters
        self._load_dc_parameters()

        # Build optimization problem (16 variables)
        self._build_optimization_problem()

        # Warm start (18 variables now)
        self.x0 = np.concatenate([
            np.ones(self.D) / self.D,  # p_cpu
            np.ones(self.D) / self.D,  # p_gpu
            [1.0],                     # rho_cpu (start with full admission)
            [1.0],                     # rho_gpu (start with full admission)
            np.full(self.D, 22.0),     # T_target
            np.zeros(self.D)           # xi (slack)
        ])

        # Performance tracking
        self.solve_times = []

        if verbose:
            print(f"[Stage1] Initialized with N={N}, D={self.D}")

    def _default_weights(self) -> Dict[str, float]:
        """Default MPC weights for Stage 1."""
        return {
            'wQ': 0.01,     # Queue minimization (reduced - don't fight utilization)
            'wT': 10.0,     # Thermal safety (high priority)
            'wE': 0.01,     # Energy cost (regularizer)
            'wL': 0.5,      # Load balancing
            'wS': 100.0,    # Slack penalty (enforce capacity)
            'wU': 0.1       # Utilization balance (REDUCED 10,000x - allow flexible utilization)
        }

    def _load_dc_parameters(self):
        """Load and aggregate DC-level thermal and capacity parameters."""
        self.dt = 300.0  # 5 minutes in seconds

        # Thermal parameters per DC
        self.C = np.zeros(self.D)  # Thermal capacitance (J/K)
        self.R = np.zeros(self.D)  # Thermal resistance (K/W)
        self.K_p = np.zeros(self.D)  # Cooling controller gain (W/K)
        self.cooling_max = np.zeros(self.D)  # Max cooling power (W)
        self.theta_ref = np.zeros(self.D)  # Reference temperature (°C)
        self.theta_max = np.zeros(self.D)  # Maximum temperature (°C)
        self.psi = np.zeros(self.D)  # Energy price ($/kWh)

        # Heat generation coefficients (average per DC)
        self.alpha_cpu_avg = np.zeros(self.D)
        self.alpha_gpu_avg = np.zeros(self.D)

        # Capacity per DC
        self.C_dc_cpu = np.zeros(self.D)  # Total CPU capacity per DC
        self.C_dc_gpu = np.zeros(self.D)  # Total GPU capacity per DC

        for d in range(self.D):
            # Get datacenter from environment
            dc = self.env.datacenters[d]
            dc_params = self.env.config['datacenter_params'][d]

            # Thermal parameters
            self.C[d] = dc_params.get('C', 1e7)  # J/K
            self.R[d] = dc_params.get('R', 0.001)  # K/W

            # Cooling controller parameters
            controller_params = dc_params.get('controller_params', {})
            self.K_p[d] = controller_params.get('K_p', 1000.0)  # W/K
            self.cooling_max[d] = dc_params.get('cooling_max', 10000.0)  # W

            # Thermal limits
            throttling_config = dc_params.get('thermal_throttling', {})
            self.theta_max[d] = dc_params.get('theta_max', 35.0)  # °C
            self.theta_ref[d] = 27.0  # Target reference temperature (balanced for 60-70% util)

            # Energy cost
            self.psi[d] = dc_params.get('psi', 0.10)  # $/kWh

            # Aggregate cluster parameters for this DC
            cpu_clusters_in_dc = []
            gpu_clusters_in_dc = []

            for i in range(self.M):
                if self.env.cluster_to_dc[i] == d:
                    cluster = self.env.clusters[i]
                    cluster_params = self.env.config['cluster_params'][i]

                    if cluster_params.get('type', 'cpu') == 'cpu':
                        cpu_clusters_in_dc.append(i)
                        self.C_dc_cpu[d] += cluster.get('c', 0)
                    else:
                        gpu_clusters_in_dc.append(i)
                        self.C_dc_gpu[d] += cluster.get('c', 0)

            # Average heat generation coefficients
            if cpu_clusters_in_dc:
                alphas = [self.env.config['cluster_params'][i].get('alpha', 10.0) for i in cpu_clusters_in_dc]
                self.alpha_cpu_avg[d] = np.mean(alphas)
            else:
                self.alpha_cpu_avg[d] = 10.0  # Default

            if gpu_clusters_in_dc:
                alphas = [self.env.config['cluster_params'][i].get('alpha', 20.0) for i in gpu_clusters_in_dc]
                self.alpha_gpu_avg[d] = np.mean(alphas)
            else:
                self.alpha_gpu_avg[d] = 20.0  # Default

        # Initialize feedback correction factors (will be adapted during operation)
        self.alpha_correction_cpu = np.ones(self.D)
        self.alpha_correction_gpu = np.ones(self.D)
        self.temp_history = []  # Store (predicted, actual) tuples for learning
        self.last_predicted_temps = None

        if self.verbose:
            print(f"[Stage1] DC capacities - CPU: {self.C_dc_cpu}, GPU: {self.C_dc_gpu}")
            print(f"[Stage1] Thermal params - C: {self.C}, R: {self.R}, K_p: {self.K_p}")

    def _compute_effective_capacity(self, cluster_ids: List[int], avg_job_size: float) -> float:
        """
        Compute effective capacity accounting for fragmentation.

        Uses a more aggressive approach: weight capacity by cluster utilization
        to penalize sending jobs to highly fragmented DCs.

        Fragmentation penalty: If cluster utilization variance is high,
        reduce effective capacity.

        Args:
            cluster_ids: List of cluster IDs to aggregate
            avg_job_size: Average job resource requirement (75th percentile for conservative estimate)

        Returns:
            Effective capacity (in resource units)
        """
        if not cluster_ids or avg_job_size <= 0:
            return 0.0

        # Compute cluster utilizations and available capacities
        utilizations = []
        availables = []

        for cluster_id in cluster_ids:
            cluster = self.env.clusters[cluster_id]
            capacity = cluster['c']
            used = cluster['u']
            available = capacity - used

            if capacity > 0:
                utilizations.append(used / capacity)
                availables.append(available)
            else:
                utilizations.append(1.0)  # Full
                availables.append(0)

        if not availables:
            return 0.0

        # AGGRESSIVE FIX: Use fragmentation-penalized capacity
        # If utilizations are highly imbalanced, reduce effective capacity
        total_available = sum(availables)

        if len(utilizations) > 1:
            max_util = max(utilizations)
            min_util = min(utilizations)
            fragmentation_gap = max_util - min_util

            # If max cluster is >80% full and there's >30% gap, heavily penalize
            if max_util > 0.80 and fragmentation_gap > 0.30:
                # Only count capacity from the LEAST utilized clusters
                # (conservative: assume jobs will hit the full cluster first)
                penalty_factor = 0.5  # Reduce effective capacity by 50%
                penalized_capacity = total_available * penalty_factor
                if self.verbose:  # Only log in verbose mode
                    print(f"  ⚠️  FRAGMENTATION PENALTY APPLIED: {int(total_available)} → {int(penalized_capacity)} (max_util={max_util:.1%}, gap={fragmentation_gap:.1%})")
                return penalized_capacity

        return total_available

    def _build_optimization_problem(self):
        """
        Build Stage 1 NLP with feasibility constraints in job units.

        Decision variables:
        - p_cpu[d]: CPU allocation ratios (4 vars)
        - p_gpu[d]: GPU allocation ratios (4 vars)
        - rho_cpu: Global CPU admission fraction (1 var)
        - rho_gpu: Global GPU admission fraction (1 var)
        - T_target[d]: Temperature setpoints (4 vars)
        - xi[d]: Thermal slack variables (4 vars)
        Total: 18 variables (was 16)
        """
        # Decision variables (18 total)
        p_cpu = ca.MX.sym('p_cpu', self.D)  # CPU allocation ratios
        p_gpu = ca.MX.sym('p_gpu', self.D)  # GPU allocation ratios
        rho_cpu = ca.MX.sym('rho_cpu', 1)   # Global CPU admission fraction [0,1]
        rho_gpu = ca.MX.sym('rho_gpu', 1)   # Global GPU admission fraction [0,1]
        T_target = ca.MX.sym('T_target', self.D)  # Temperature setpoints
        xi = ca.MX.sym('xi', self.D)  # Thermal slack variables

        # Parameters (current state and job info)
        theta_current = ca.MX.sym('theta', self.D)
        q_cpu_current = ca.MX.sym('q_cpu', self.D)
        q_gpu_current = ca.MX.sym('q_gpu', self.D)
        u_cpu_current = ca.MX.sym('u_cpu', self.D)
        u_gpu_current = ca.MX.sym('u_gpu', self.D)
        n_cpu = ca.MX.sym('n_cpu', 1)  # Number of CPU jobs arriving
        n_gpu = ca.MX.sym('n_gpu', 1)  # Number of GPU jobs arriving
        T_amb = ca.MX.sym('T_amb', self.D)
        r_bar_cpu = ca.MX.sym('r_bar_cpu', 1)  # Average CPU job resource
        r_bar_gpu = ca.MX.sym('r_bar_gpu', 1)  # Average GPU job resource

        # NEW: Max feasible jobs per DC (computed from residual headroom / job size)
        Jmax_cpu = ca.MX.sym('Jmax_cpu', self.D)  # Max CPU jobs feasible per DC
        Jmax_gpu = ca.MX.sym('Jmax_gpu', self.D)  # Max GPU jobs feasible per DC

        # Constraints
        g = []
        lbg = []
        ubg = []

        # 1. NEW: Feasibility-aware allocation constraints (replaces simplex)
        # sum(p_cpu) = rho_cpu (fraction of CPU jobs to attempt admitting)
        g.append(ca.sum1(p_cpu) - rho_cpu)
        lbg.append(0.0)
        ubg.append(0.0)

        g.append(ca.sum1(p_gpu) - rho_gpu)
        lbg.append(0.0)
        ubg.append(0.0)

        # 2. NEW: Per-DC feasibility constraints (p[d] * n <= Jmax[d])
        for d in range(self.D):
            # CPU jobs allocated to DC d must not exceed max feasible
            g.append(p_cpu[d] * n_cpu - Jmax_cpu[d])
            lbg.append(-ca.inf)
            ubg.append(0.0)  # p_cpu[d] * n_cpu <= Jmax_cpu[d]

            # GPU jobs allocated to DC d must not exceed max feasible
            g.append(p_gpu[d] * n_gpu - Jmax_gpu[d])
            lbg.append(-ca.inf)
            ubg.append(0.0)  # p_gpu[d] * n_gpu <= Jmax_gpu[d]

        # 3. NEW: Penalize rejection (encourage rho -> 1 when feasible)
        # Weight should be comparable to rejection penalty in Stage 2
        wRho = 10000.0  # Penalty for rejecting jobs globally (INCREASED 10x for max throughput)
        obj_rejection = wRho * (n_cpu * (1.0 - rho_cpu) + n_gpu * (1.0 - rho_gpu))

        # Prediction over horizon
        obj = obj_rejection  # Start with rejection penalty
        theta = theta_current

        for k in range(self.N):
            # Incremental utilization from new jobs (scaled by admission fraction)
            delta_u_cpu = p_cpu * rho_cpu * n_cpu * r_bar_cpu
            delta_u_gpu = p_gpu * rho_gpu * n_gpu * r_bar_gpu

            # Total predicted utilization (with decay over horizon)
            decay_factor = 1.0 - k * 0.2  # Decay as jobs complete
            u_pred_cpu = u_cpu_current + delta_u_cpu * decay_factor
            u_pred_gpu = u_gpu_current + delta_u_gpu * decay_factor

            # Heat generation per DC (with feedback-corrected alphas)
            alpha_cpu_corrected = self.alpha_cpu_avg * self.alpha_correction_cpu
            alpha_gpu_corrected = self.alpha_gpu_avg * self.alpha_correction_gpu
            Q_gen = alpha_cpu_corrected * u_pred_cpu + alpha_gpu_corrected * u_pred_gpu

            # Cooling and thermal dynamics
            theta_next = ca.MX.zeros(self.D)
            Q_cool = ca.MX.zeros(self.D)

            for d in range(self.D):
                # Proportional cooling with optimized T_target
                Q_cool[d] = self.K_p[d] * ca.fmax(0, theta[d] - T_target[d])
                Q_cool[d] = ca.fmin(Q_cool[d], self.cooling_max[d])

                # RC thermal model
                dtheta = (self.dt / self.C[d]) * (
                    Q_gen[d] - (theta[d] - T_amb[d]) / self.R[d] - Q_cool[d]
                )
                theta_next[d] = theta[d] + dtheta

            # HARD TEMPERATURE CONSTRAINTS at 31°C (theta_ref=27°C is soft penalty)
            for d in range(self.D):
                g.append(theta_next[d])
                lbg.append(-ca.inf)
                ubg.append(31.0)  # Hard limit with feedback correction

            # Objective components

            # 1. Minimize current queues (existing backlog)
            obj += self.weights['wQ'] * (ca.sum1(q_cpu_current**2) + ca.sum1(q_gpu_current**2))

            # 2. Thermal penalty (safety)
            for d in range(self.D):
                temp_violation = ca.fmax(0, theta_next[d] - self.theta_ref[d])
                obj += self.weights['wT'] * temp_violation**2

            # 3. Energy cost
            for d in range(self.D):
                energy_kwh = Q_cool[d] * self.dt / 3.6e6  # Convert J to kWh
                obj += self.weights['wE'] * self.psi[d] * energy_kwh

            # 4. Load balancing (minimize variance across DCs)
            u_total = u_pred_cpu + u_pred_gpu
            u_mean = ca.sum1(u_total) / self.D
            for d in range(self.D):
                obj += self.weights['wL'] * (u_total[d] - u_mean)**2

            # 5. Utilization target penalty (encourage 60-70% utilization)
            # Target: 0.65 (65% utilization)
            target_util = 0.65
            for d in range(self.D):
                util_cpu_ratio = u_pred_cpu[d] / (self.C_dc_cpu[d] + 1e-6)  # Avoid div by zero
                util_gpu_ratio = u_pred_gpu[d] / (self.C_dc_gpu[d] + 1e-6)

                # Penalize deviation from target (both under and over)
                obj += self.weights['wU'] * (util_cpu_ratio - target_util)**2
                obj += self.weights['wU'] * (util_gpu_ratio - target_util)**2

            # Update state for next timestep
            theta = theta_next

        # 5. Slack penalty (enforce thermal limits via soft constraints)
        obj += self.weights['wS'] * ca.sum1(xi)

        # Create NLP
        x = ca.vertcat(p_cpu, p_gpu, rho_cpu, rho_gpu, T_target, xi)
        p = ca.vertcat(theta_current, q_cpu_current, q_gpu_current,
                       u_cpu_current, u_gpu_current, n_cpu, n_gpu, T_amb,
                       r_bar_cpu, r_bar_gpu, Jmax_cpu, Jmax_gpu)

        nlp = {'x': x, 'f': obj, 'g': ca.vertcat(*g), 'p': p}

        # Solver options
        opts = {
            'ipopt.print_level': 0 if not self.verbose else 5,
            'ipopt.max_iter': 50,  # Fast convergence
            'ipopt.tol': 1e-4,     # Balance speed/accuracy
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_strategy': 'adaptive',
            'print_time': 0 if not self.verbose else 1
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Store constraint bounds
        self.lbg = lbg
        self.ubg = ubg

        if self.verbose:
            print(f"[Stage1] MPC problem built. Variables: 16, Constraints: {len(lbg)}")

    def solve(self, state: Dict, jobs: List[Dict]) -> Dict[str, Any]:
        """
        Solve Stage 1 DC-level optimization with temperature feedback correction.

        Args:
            state: Current system state
            jobs: List of arriving jobs

        Returns:
            Dict with:
                - ratios: {'cpu': p_cpu, 'gpu': p_gpu}
                - T_target: Temperature setpoints [D]
                - job_counts: {'cpu': n_cpu_per_dc, 'gpu': n_gpu_per_dc}
        """
        start_time = time.time()

        # FEEDBACK CORRECTION: Update alpha coefficients based on historical prediction errors
        if len(self.temp_history) > 0:
            # Simple adaptive correction: if we underpredicted heat (actual > predicted),
            # increase alpha for next time
            recent_errors = self.temp_history[-5:]  # Last 5 timesteps
            for d in range(self.D):
                avg_error = np.mean([actual[d] - pred[d] for pred, actual in recent_errors])
                if avg_error > 0:  # Underpredicting temperature (generating more heat than expected)
                    # Increase alpha correction by 5% of error
                    self.alpha_correction_cpu[d] *= (1.0 + 0.05 * min(avg_error / 10.0, 0.5))
                    self.alpha_correction_gpu[d] *= (1.0 + 0.05 * min(avg_error / 10.0, 0.5))
                elif avg_error < -1.0:  # Overpredicting significantly
                    # Decrease alpha correction
                    self.alpha_correction_cpu[d] *= (1.0 + 0.05 * max(avg_error / 10.0, -0.3))
                    self.alpha_correction_gpu[d] *= (1.0 + 0.05 * max(avg_error / 10.0, -0.3))

        # EXPERIMENT 3: Bypass Stage 1 optimization - use uniform allocation
        EXPERIMENT_3_ENABLED = False  # Disabled - testing Stage 2 fixes

        if EXPERIMENT_3_ENABLED:
            # Count jobs by type
            cpu_jobs = [j for j in jobs if j.get('tau', 'cpu') == 'cpu']
            gpu_jobs = [j for j in jobs if j.get('tau', 'cpu') == 'gpu']
            n_cpu = len(cpu_jobs)
            n_gpu = len(gpu_jobs)

            print(f"[EXPERIMENT 3] Bypassing Stage 1 - using uniform allocation")
            print(f"[EXPERIMENT 3] Distributing {n_cpu} CPU, {n_gpu} GPU jobs uniformly")

            # Uniform allocation - 25% to each DC
            return {
                'ratios': {
                    'cpu': np.ones(self.D) / self.D,
                    'gpu': np.ones(self.D) / self.D
                },
                'T_target': np.full(self.D, 22.0),
                'job_counts': {
                    'cpu': self._round_preserving_sum(np.ones(self.D) * n_cpu / self.D),
                    'gpu': self._round_preserving_sum(np.ones(self.D) * n_gpu / self.D)
                }
            }

        # Count jobs by type FIRST (needed for effective capacity computation)
        cpu_jobs = [j for j in jobs if j.get('tau', 'cpu') == 'cpu']
        gpu_jobs = [j for j in jobs if j.get('tau', 'cpu') == 'gpu']

        n_cpu = len(cpu_jobs)
        n_gpu = len(gpu_jobs)

        # FIX OPTION A: Compute effective capacity accounting for fragmentation
        # Use 50th percentile (median) for balanced admission targeting 70% utilization
        if cpu_jobs:
            cpu_job_sizes = [j['r'] for j in cpu_jobs]
            r_eff_cpu = np.percentile(cpu_job_sizes, 50)
        else:
            r_eff_cpu = 1000.0  # Default

        if gpu_jobs:
            gpu_job_sizes = [j['r'] for j in gpu_jobs]
            r_eff_gpu = np.percentile(gpu_job_sizes, 50)
        else:
            r_eff_gpu = 5000.0  # Default

        # CRITICAL FIX: Re-read current capacity values from environment
        # AND compute effective capacity accounting for cluster-level fragmentation
        self.C_dc_cpu = np.zeros(self.D)
        self.C_dc_gpu = np.zeros(self.D)

        for d in range(self.D):
            # Get cluster IDs for this DC
            cpu_clusters_in_dc = []
            gpu_clusters_in_dc = []

            for i in range(self.env.M):
                if self.env.cluster_to_dc[i] == d:
                    cluster_params = self.env.config['cluster_params'][i]
                    if cluster_params.get('type', 'cpu') == 'cpu':
                        cpu_clusters_in_dc.append(i)
                    else:
                        gpu_clusters_in_dc.append(i)

            # Compute EFFECTIVE capacity (accounts for fragmentation)
            self.C_dc_cpu[d] = self._compute_effective_capacity(cpu_clusters_in_dc, r_eff_cpu)
            self.C_dc_gpu[d] = self._compute_effective_capacity(gpu_clusters_in_dc, r_eff_gpu)

        if self.verbose:
            print(f"[Stage1] Solving for {n_cpu} CPU, {n_gpu} GPU jobs")
            print(f"[Stage1] Effective DC capacities - CPU: {self.C_dc_cpu}, GPU: {self.C_dc_gpu}")

        # FIX OPTION A: Add diagnostics to compare effective vs aggregate capacity
        if self.verbose:  # Disabled for clean full run
            # Compute aggregate capacity for comparison
            C_dc_cpu_aggregate = np.zeros(self.D)
            C_dc_gpu_aggregate = np.zeros(self.D)

            for d in range(self.D):
                for i in range(self.env.M):
                    if self.env.cluster_to_dc[i] == d:
                        cluster = self.env.clusters[i]
                        cluster_params = self.env.config['cluster_params'][i]
                        available = cluster['c'] - cluster['u']
                        if cluster_params.get('type', 'cpu') == 'cpu':
                            C_dc_cpu_aggregate[d] += available
                        else:
                            C_dc_gpu_aggregate[d] += available

            print("\n=== FIX OPTION A: EFFECTIVE vs AGGREGATE CAPACITY ===")
            for d in range(self.D):
                if C_dc_cpu_aggregate[d] > 0:
                    eff_ratio_cpu = self.C_dc_cpu[d] / C_dc_cpu_aggregate[d]
                    print(f"DC{d} CPU: Effective={int(self.C_dc_cpu[d])}, Aggregate={int(C_dc_cpu_aggregate[d])}, Ratio={eff_ratio_cpu:.2f}")
                if C_dc_gpu_aggregate[d] > 0:
                    eff_ratio_gpu = self.C_dc_gpu[d] / C_dc_gpu_aggregate[d]
                    print(f"DC{d} GPU: Effective={int(self.C_dc_gpu[d])}, Aggregate={int(C_dc_gpu_aggregate[d])}, Ratio={eff_ratio_gpu:.2f}")
            print()

        # Handle edge case: no jobs
        if n_cpu == 0 and n_gpu == 0:
            return {
                'ratios': {
                    'cpu': np.ones(self.D) / self.D,
                    'gpu': np.ones(self.D) / self.D
                },
                'T_target': np.full(self.D, 22.0),
                'job_counts': {
                    'cpu': np.zeros(self.D, dtype=int),
                    'gpu': np.zeros(self.D, dtype=int)
                }
            }

        # Calculate average resource demands
        r_bar_cpu = np.mean([j['r'] for j in cpu_jobs]) if cpu_jobs else 1000.0
        r_bar_gpu = np.mean([j['r'] for j in gpu_jobs]) if gpu_jobs else 5000.0

        # Use 50th percentile (median) for balanced admission targeting 70% utilization
        if cpu_jobs:
            r_eff_cpu = np.percentile([j['r'] for j in cpu_jobs], 50)
        else:
            r_eff_cpu = r_bar_cpu

        if gpu_jobs:
            r_eff_gpu = np.percentile([j['r'] for j in gpu_jobs], 50)
        else:
            r_eff_gpu = r_bar_gpu

        # NEW: Compute Jmax (max feasible jobs per DC) in job units
        Jmax_cpu = np.zeros(self.D)
        Jmax_gpu = np.zeros(self.D)

        for d in range(self.D):
            # Get cluster IDs for this DC
            cpu_clusters_in_dc = []
            gpu_clusters_in_dc = []

            for i in range(self.env.M):
                if self.env.cluster_to_dc[i] == d:
                    cluster_params = self.env.config['cluster_params'][i]
                    if cluster_params.get('type', 'cpu') == 'cpu':
                        cpu_clusters_in_dc.append(i)
                    else:
                        gpu_clusters_in_dc.append(i)

            # Sum per-cluster max feasible jobs
            for cluster_id in cpu_clusters_in_dc:
                cluster = self.env.clusters[cluster_id]
                residual = max(0.0, cluster.get('c', 0) - cluster.get('u', 0))
                max_jobs = int(np.floor(residual / r_eff_cpu)) if r_eff_cpu > 0 else 0
                Jmax_cpu[d] += max_jobs

            for cluster_id in gpu_clusters_in_dc:
                cluster = self.env.clusters[cluster_id]
                residual = max(0.0, cluster.get('c', 0) - cluster.get('u', 0))
                max_jobs = int(np.floor(residual / r_eff_gpu)) if r_eff_gpu > 0 else 0
                Jmax_gpu[d] += max_jobs

        if self.verbose:
            print(f"[Stage1] Jmax_cpu per DC: {Jmax_cpu} (total: {np.sum(Jmax_cpu):.0f} vs arrivals: {n_cpu})")
            print(f"[Stage1] Jmax_gpu per DC: {Jmax_gpu} (total: {np.sum(Jmax_gpu):.0f} vs arrivals: {n_gpu})")

        # Pack parameters (now includes Jmax)
        p_values = np.concatenate([
            state['theta'],
            state['q_cpu'],
            state['q_gpu'],
            state['u_cpu'],
            state['u_gpu'],
            [n_cpu],
            [n_gpu],
            state['T_amb'],
            [r_bar_cpu],
            [r_bar_gpu],
            Jmax_cpu,
            Jmax_gpu
        ])

        # Setup bounds (18 variables now)
        lbx = ([0.0] * self.D +       # p_cpu >= 0
               [0.0] * self.D +       # p_gpu >= 0
               [0.0] +                # rho_cpu >= 0
               [0.0] +                # rho_gpu >= 0
               [18.0] * self.D +      # T_target >= 18°C
               [0.0] * self.D)        # xi >= 0

        ubx = ([1.0] * self.D +       # p_cpu <= 1
               [1.0] * self.D +       # p_gpu <= 1
               [1.0] +                # rho_cpu <= 1
               [1.0] +                # rho_gpu <= 1
               [28.0] * self.D +      # T_target <= 28°C
               [np.inf] * self.D)     # xi unbounded above

        # Solve
        try:
            sol = self.solver(
                x0=self.x0,
                lbx=lbx,
                ubx=ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p_values
            )

            # Extract solution (18 variables now)
            x_opt = sol['x'].full().flatten()

            p_cpu = x_opt[:self.D]
            p_gpu = x_opt[self.D:2*self.D]
            rho_cpu = float(x_opt[2*self.D])
            rho_gpu = float(x_opt[2*self.D + 1])
            T_target = x_opt[2*self.D+2:3*self.D+2]
            xi = x_opt[3*self.D+2:]

            # Update warm start
            self.x0 = x_opt

            # Normalize ratios to match rho (handle numerical errors)
            if np.sum(p_cpu) > 0:
                p_cpu = p_cpu / np.sum(p_cpu) * rho_cpu
            else:
                p_cpu = np.ones(self.D) / self.D * rho_cpu

            if np.sum(p_gpu) > 0:
                p_gpu = p_gpu / np.sum(p_gpu) * rho_gpu
            else:
                p_gpu = np.ones(self.D) / self.D * rho_gpu

            # Convert ratios to job counts (scaled by admission fraction rho)
            job_counts_cpu = self._round_preserving_sum(p_cpu * n_cpu)
            job_counts_gpu = self._round_preserving_sum(p_gpu * n_gpu)

            solve_time = time.time() - start_time
            self.solve_times.append(solve_time)

            if self.verbose:
                print(f"[Stage1] Solved in {solve_time*1000:.1f}ms")
                print(f"[Stage1] Admission fractions: rho_cpu={rho_cpu:.3f}, rho_gpu={rho_gpu:.3f}")
                print(f"[Stage1] Jobs admitted: CPU={int(np.sum(job_counts_cpu))}/{n_cpu}, GPU={int(np.sum(job_counts_gpu))}/{n_gpu}")
                print(f"[Stage1] CPU ratios: {p_cpu}")
                print(f"[Stage1] GPU ratios: {p_gpu}")
                print(f"[Stage1] T_target: {T_target}")
                print(f"[Stage1] Slack: {xi}")

            # EXPERIMENT 1: Stage 1 Allocation Diagnostics
            if self.verbose:  # Disabled for Experiment 3
                print("\n=== STAGE 1 ALLOCATION DIAGNOSTICS ===")
                for d in range(self.D):
                    # Get cluster IDs for this DC
                    cluster_ids = [i for i in range(self.env.M) if self.env.cluster_to_dc[i] == d]
                    cpu_clusters = [i for i in cluster_ids
                                   if self.env.config['cluster_params'][i].get('type', 'cpu') == 'cpu']
                    gpu_clusters = [i for i in cluster_ids
                                   if self.env.config['cluster_params'][i].get('type', 'cpu') == 'gpu']

                    # CPU clusters diagnostics
                    if cpu_clusters and job_counts_cpu[d] > 0:
                        cluster_utils = [self.env.clusters[i]['u'] / (self.env.clusters[i]['c'] + 1e-6)
                                       for i in cpu_clusters]
                        cluster_free = [self.env.clusters[i]['c'] - self.env.clusters[i]['u']
                                      for i in cpu_clusters]

                        dc_agg_util = state['u_cpu'][d] / (self.C_dc_cpu[d] + 1e-6)

                        print(f"DC{d} CPU: Stage 1 allocated {int(job_counts_cpu[d])} jobs")
                        print(f"  Cluster utils: {[f'{u:.1%}' for u in cluster_utils]}")
                        print(f"  Cluster free:  {[int(f) for f in cluster_free]}")
                        print(f"  DC aggregate:  {dc_agg_util:.1%}")

                        # Highlight fragmentation
                        if len(cluster_utils) > 1:
                            max_util = max(cluster_utils)
                            min_util = min(cluster_utils)
                            if max_util > 0.85 and min_util < 0.5:
                                print(f"  ⚠️  FRAGMENTATION DETECTED: {max_util:.1%} / {min_util:.1%}")

                    # GPU clusters diagnostics
                    if gpu_clusters and job_counts_gpu[d] > 0:
                        cluster_utils = [self.env.clusters[i]['u'] / (self.env.clusters[i]['c'] + 1e-6)
                                       for i in gpu_clusters]
                        cluster_free = [self.env.clusters[i]['c'] - self.env.clusters[i]['u']
                                      for i in gpu_clusters]

                        dc_agg_util = state['u_gpu'][d] / (self.C_dc_gpu[d] + 1e-6)

                        print(f"DC{d} GPU: Stage 1 allocated {int(job_counts_gpu[d])} jobs")
                        print(f"  Cluster utils: {[f'{u:.1%}' for u in cluster_utils]}")
                        print(f"  Cluster free:  {[int(f) for f in cluster_free]}")
                        print(f"  DC aggregate:  {dc_agg_util:.1%}")

                        # Highlight fragmentation
                        if len(cluster_utils) > 1:
                            max_util = max(cluster_utils)
                            min_util = min(cluster_utils)
                            if max_util > 0.85 and min_util < 0.5:
                                print(f"  ⚠️  FRAGMENTATION DETECTED: {max_util:.1%} / {min_util:.1%}")

            # Store predicted temperature for feedback learning
            # Compute predicted temp at next timestep for comparison
            delta_u_cpu_pred = p_cpu * n_cpu * r_bar_cpu
            delta_u_gpu_pred = p_gpu * n_gpu * r_bar_gpu
            u_pred_cpu_next = state['u_cpu'] + delta_u_cpu_pred
            u_pred_gpu_next = state['u_gpu'] + delta_u_gpu_pred
            Q_gen_pred = alpha_cpu_corrected * u_pred_cpu_next + alpha_gpu_corrected * u_pred_gpu_next

            # Simple 1-step thermal prediction
            theta_pred = state['theta'].copy()
            for d in range(self.D):
                Q_cool_pred = self.K_p[d] * max(0, state['theta'][d] - T_target[d])
                Q_cool_pred = min(Q_cool_pred, self.cooling_max[d])
                dtheta = (self.dt / self.C[d]) * (
                    Q_gen_pred[d] - (state['theta'][d] - state['T_amb'][d]) / self.R[d] - Q_cool_pred
                )
                theta_pred[d] = state['theta'][d] + dtheta

            self.last_predicted_temps = theta_pred

            # Store history for learning (keep last 10 entries)
            if len(self.temp_history) >= 10:
                self.temp_history.pop(0)
            self.temp_history.append((theta_pred.copy(), state['theta'].copy()))

            return {
                'ratios': {'cpu': p_cpu, 'gpu': p_gpu},
                'T_target': T_target,
                'job_counts': {'cpu': job_counts_cpu, 'gpu': job_counts_gpu}
            }

        except Exception as e:
            if self.verbose:
                print(f"[Stage1] Solver failed: {e}")

            # Fallback: uniform allocation
            return {
                'ratios': {
                    'cpu': np.ones(self.D) / self.D,
                    'gpu': np.ones(self.D) / self.D
                },
                'T_target': np.full(self.D, 22.0),
                'job_counts': {
                    'cpu': self._round_preserving_sum(np.ones(self.D) * n_cpu / self.D),
                    'gpu': self._round_preserving_sum(np.ones(self.D) * n_gpu / self.D)
                }
            }

    def _round_preserving_sum(self, x: np.ndarray) -> np.ndarray:
        """Round array to integers while preserving sum."""
        rounded = np.floor(x).astype(int)
        remainder = int(np.round(np.sum(x) - np.sum(rounded)))

        if remainder > 0:
            frac = x - rounded
            indices = np.argsort(-frac)[:remainder]
            rounded[indices] += 1

        return rounded


class HierarchicalMPCStage2:
    """
    Stage 2: Per-DC Cluster-Level MPC Optimization.

    For each DC, optimizes job-to-cluster assignments given:
    - Jobs allocated to this DC by Stage 1
    - Temperature setpoint from Stage 1

    Decision variables: x[j,i] for job-cluster assignments (~180 vars per DC)
    """

    def __init__(self, env, dc_id: int, N: int = 3,
                 weights: Optional[Dict[str, float]] = None,
                 verbose: bool = False):
        """
        Initialize Stage 2 per-DC cluster MPC with aggregate flows.

        Args:
            env: DataCenterEnv instance
            dc_id: Datacenter ID this solver is for
            N: MPC horizon length (3 timesteps = 15 min)
            weights: Dict with keys: wQ, wT, wE, wU, wR, wC, wSmooth
            verbose: Print debug information
        """
        self.env = env
        self.dc_id = dc_id
        self.N = N
        self.weights = weights or self._default_weights()
        self.verbose = verbose

        # Get clusters in this DC
        self.cluster_ids = [i for i in range(env.M) if env.cluster_to_dc[i] == dc_id]
        self.M_dc = len(self.cluster_ids)  # Number of clusters in this DC

        # Load parameters
        self._load_parameters()

        # Build optimization problem (this sets self.n_cpu_clusters, self.n_gpu_clusters)
        self._build_optimization_problem()

        # Warm start (computed after _build_optimization_problem)
        n_vars = 4*N + (self.n_cpu_clusters + self.n_gpu_clusters)*N + N
        self.x0 = np.zeros(n_vars)

        # Performance tracking
        self.solve_times = []

        if verbose:
            print(f"[Stage2-DC{dc_id}] Initialized with N={N}, M_dc={self.M_dc}")

    def _default_weights(self) -> Dict[str, float]:
        """Optimized MPC weights for maximum throughput with thermal safety."""
        return {
            'wQ': 1.0,       # Queue minimization
            'wT': 5.0,       # Thermal penalty (lower priority since Stage 1 handles it)
            'wE': 0.01,      # Energy cost (linear in allocation)
            'wU': 0.1,       # Utilization balance (REDUCED 5x to allow imbalance)
            'wR': 15000.0,   # Rejection penalty (INCREASED 3x to maximize throughput)
            'wC': 500.0,     # Capacity slack penalty (INCREASED 50x - Phase 3 fix)
            'wSmooth': 0.0   # Smoothness penalty (DISABLED to allow aggressive admission - was 0.5)
        }

    def _load_parameters(self):
        """Load DC and cluster parameters."""
        self.dt = 300.0  # 5 minutes in seconds

        # DC thermal parameters
        dc_params = self.env.config['datacenter_params'][self.dc_id]
        self.C = dc_params.get('C', 1e7)
        self.R = dc_params.get('R', 0.001)

        controller_params = dc_params.get('controller_params', {})
        self.K_p = controller_params.get('K_p', 1000.0)
        self.cooling_max = dc_params.get('cooling_max', 10000.0)

        throttling_config = dc_params.get('thermal_throttling', {})
        self.theta_ref = 27.0  # Target reference temperature (balanced for 60-70% util)

        # Cluster parameters
        self.alpha = np.zeros(self.M_dc)
        self.c_max = np.zeros(self.M_dc)
        self.cluster_types = []

        for idx, cluster_id in enumerate(self.cluster_ids):
            cluster_params = self.env.config['cluster_params'][cluster_id]
            self.alpha[idx] = cluster_params.get('alpha', 10.0)
            self.c_max[idx] = cluster_params.get('c_max', cluster_params.get('c', 10000))
            self.cluster_types.append(cluster_params.get('type', 'cpu'))

        if self.verbose:
            print(f"[Stage2-DC{self.dc_id}] Cluster capacities: {self.c_max}")
            print(f"[Stage2-DC{self.dc_id}] Heat coefficients: {self.alpha}")

    def _build_optimization_problem(self):
        """
        Build Stage 2 NLP for per-DC cluster assignment using aggregate flow variables.

        Decision variables (aggregate flows, NOT job-indexed):
        - a_cpu[k], a_gpu[k]: Admitted CPU/GPU jobs at timestep k
        - rj_cpu[k], rj_gpu[k]: Rejected CPU/GPU jobs at timestep k
        - x_cpu[c,k], x_gpu[c,k]: Flow allocated to cluster c at timestep k
        - slack_theta[k]: Thermal slack variable

        Scales as O(M_dc × N) instead of O(n_jobs × M_dc)
        """
        M_dc = self.M_dc
        N = self.N

        # Count CPU and GPU clusters
        n_cpu_clusters = sum(1 for t in self.cluster_types if t == 'cpu')
        n_gpu_clusters = sum(1 for t in self.cluster_types if t == 'gpu')

        # ===== PHASE 1: DECISION VARIABLES =====
        # Aggregate flow variables (per timestep in horizon)
        a_cpu = ca.MX.sym('a_cpu', N)     # Admitted CPU jobs
        a_gpu = ca.MX.sym('a_gpu', N)     # Admitted GPU jobs
        rj_cpu = ca.MX.sym('rj_cpu', N)   # Rejected CPU jobs
        rj_gpu = ca.MX.sym('rj_gpu', N)   # Rejected GPU jobs

        # Cluster flow allocation (per timestep)
        x_cpu = ca.MX.sym('x_cpu', n_cpu_clusters * N)   # CPU cluster flows
        x_gpu = ca.MX.sym('x_gpu', n_gpu_clusters * N)   # GPU cluster flows

        # Thermal slack (keep for soft constraints)
        slack_theta = ca.MX.sym('slack_theta', N)

        # Total variables: 2N + 2N + 2N*M_dc + N = N(4 + 2M_dc + 1)
        # For N=3, M_dc=6: 3(4 + 12 + 1) = 51 variables vs 356 before

        # ===== PARAMETERS =====
        p_theta = ca.MX.sym('p_theta', 1)              # Current DC temperature
        p_u = ca.MX.sym('p_u', M_dc)                   # Current cluster utilizations
        p_q = ca.MX.sym('p_q', M_dc)                   # Current cluster queues
        p_T_amb = ca.MX.sym('p_T_amb', 1)              # Ambient temperature
        p_T_target = ca.MX.sym('p_T_target', 1)        # From Stage 1

        # NEW: Aggregate job arrivals per timestep
        p_J_cpu_in = ca.MX.sym('p_J_cpu_in', N)        # CPU jobs arriving each step
        p_J_gpu_in = ca.MX.sym('p_J_gpu_in', N)        # GPU jobs arriving each step
        p_r_bar_cpu = ca.MX.sym('p_r_bar_cpu', 1)      # Average CPU job size
        p_r_bar_gpu = ca.MX.sym('p_r_bar_gpu', 1)      # Average GPU job size

        # Cluster capacities (time-varying if throttling)
        p_c_cpu = ca.MX.sym('p_c_cpu', n_cpu_clusters) # CPU cluster capacities
        p_c_gpu = ca.MX.sym('p_c_gpu', n_gpu_clusters) # GPU cluster capacities

        # ===== PHASE 2: CONSTRAINTS =====
        g = []
        lbg = []
        ubg = []

        for k in range(N):
            # 2.1 Admission balance: admitted + rejected = arriving
            g.append(a_cpu[k] + rj_cpu[k] - p_J_cpu_in[k])
            lbg.append(0.0)
            ubg.append(0.0)

            g.append(a_gpu[k] + rj_gpu[k] - p_J_gpu_in[k])
            lbg.append(0.0)
            ubg.append(0.0)

            # 2.2 Flow conservation: sum of CPU cluster flows = admitted CPU jobs
            cpu_flow_sum = ca.sum1(x_cpu[k*n_cpu_clusters:(k+1)*n_cpu_clusters])
            g.append(cpu_flow_sum - a_cpu[k])
            lbg.append(0.0)
            ubg.append(0.0)

            # GPU flow conservation
            gpu_flow_sum = ca.sum1(x_gpu[k*n_gpu_clusters:(k+1)*n_gpu_clusters])
            g.append(gpu_flow_sum - a_gpu[k])
            lbg.append(0.0)
            ubg.append(0.0)

            # FIX B: Capacity constraints with correct units (jobs × job_size ≤ residual_capacity)
            # x_cpu[idx] is number of jobs, p_r_bar_cpu is avg job size in resource units
            # p_c_cpu[i] is residual capacity in resource units (from Fix A)
            for i in range(n_cpu_clusters):
                idx = k * n_cpu_clusters + i
                # Resource consumption (jobs × size) must not exceed residual capacity
                g.append(x_cpu[idx] * p_r_bar_cpu - p_c_cpu[i])
                lbg.append(-ca.inf)
                ubg.append(0.0)  # x_cpu[i,k] * r_bar_cpu ≤ c_cpu[i] (residual)

            for i in range(n_gpu_clusters):
                idx = k * n_gpu_clusters + i
                # Resource consumption (jobs × size) must not exceed residual capacity
                g.append(x_gpu[idx] * p_r_bar_gpu - p_c_gpu[i])
                lbg.append(-ca.inf)
                ubg.append(0.0)  # x_gpu[i,k] * r_bar_gpu ≤ c_gpu[i] (residual)

        # ===== PHASE 3: OBJECTIVE FUNCTION =====
        obj = 0.0
        theta = [p_theta]

        for k in range(N):
            # Compute utilization from cluster flows
            u_predicted = ca.MX.zeros(M_dc)

            cpu_cluster_idx = 0
            gpu_cluster_idx = 0

            for i in range(M_dc):
                if self.cluster_types[i] == 'cpu':
                    idx = k * n_cpu_clusters + cpu_cluster_idx
                    u_predicted[i] = p_u[i] + x_cpu[idx] * p_r_bar_cpu
                    cpu_cluster_idx += 1
                else:
                    idx = k * n_gpu_clusters + gpu_cluster_idx
                    u_predicted[i] = p_u[i] + x_gpu[idx] * p_r_bar_gpu
                    gpu_cluster_idx += 1

            # 3.1 Queue penalty (static for now)
            obj += self.weights['wQ'] * ca.sum1(p_q**2)

            # 3.2 Energy penalty (linear in allocation)
            psi = 0.10  # Default energy cost coefficient
            cpu_cluster_idx = 0
            gpu_cluster_idx = 0

            for i in range(M_dc):
                phi = self.alpha[i]  # Power coefficient

                if self.cluster_types[i] == 'cpu':
                    idx = k * n_cpu_clusters + cpu_cluster_idx
                    energy_cost = psi * phi * x_cpu[idx] * p_r_bar_cpu
                    obj += self.weights['wE'] * energy_cost
                    cpu_cluster_idx += 1
                else:
                    idx = k * n_gpu_clusters + gpu_cluster_idx
                    energy_cost = psi * phi * x_gpu[idx] * p_r_bar_gpu
                    obj += self.weights['wE'] * energy_cost
                    gpu_cluster_idx += 1

            # 3.3 Thermal penalty (keep existing model)
            # Heat generation
            Q_gen = 0.0
            for i in range(M_dc):
                Q_gen += self.alpha[i] * u_predicted[i]

            # Cooling with T_target from Stage 1
            error = ca.fmax(0, theta[k] - p_T_target)
            Q_cooling = self.K_p * error
            Q_cooling = ca.fmin(Q_cooling, self.cooling_max)

            # RC thermal dynamics
            dtheta = (self.dt / self.C) * (
                Q_gen - (theta[k] - p_T_amb) / self.R - Q_cooling
            )
            theta_next = theta[k] + dtheta
            theta.append(theta_next)

            # HARD TEMPERATURE CONSTRAINT at 31°C (theta_ref=27°C is soft penalty)
            g.append(theta_next)
            lbg.append(-ca.inf)
            ubg.append(31.0)  # Hard limit with feedback correction

            # Thermal penalty
            temp_violation = ca.fmax(0, theta_next - self.theta_ref)
            obj += self.weights['wT'] * temp_violation**2

            # 3.4 Utilization balance - COMMENTED OUT (Throughput-First Objective)
            # This term fights optimal packing and increases rejection
            # u_ratios = ca.MX.zeros(M_dc)
            # for i in range(M_dc):
            #     u_ratios[i] = u_predicted[i] / (self.c_max[i] + 1e-6)
            #
            # u_mean = ca.sum1(u_ratios) / M_dc
            # for i in range(M_dc):
            #     obj += self.weights['wU'] * (u_ratios[i] - u_mean)**2

            # 3.5 Smoothness penalty (new)
            if k > 0:
                # Penalize changes in cluster flows between timesteps
                x_cpu_prev = x_cpu[(k-1)*n_cpu_clusters:k*n_cpu_clusters]
                x_cpu_curr = x_cpu[k*n_cpu_clusters:(k+1)*n_cpu_clusters]
                obj += self.weights['wSmooth'] * ca.sum1((x_cpu_curr - x_cpu_prev)**2)

                x_gpu_prev = x_gpu[(k-1)*n_gpu_clusters:k*n_gpu_clusters]
                x_gpu_curr = x_gpu[k*n_gpu_clusters:(k+1)*n_gpu_clusters]
                obj += self.weights['wSmooth'] * ca.sum1((x_gpu_curr - x_gpu_prev)**2)

        # 3.6 NEW: Throughput reward (maximize job admission at first timestep)
        # Explicitly incentivize admitting jobs to reduce rejection rate
        wServe = 1000.0
        obj += -wServe * (a_cpu[0] + a_gpu[0])  # Negative = reward

        # 3.7 Rejection penalty (CRITICAL - sum over horizon) [was 3.6]
        for k in range(N):
            obj += self.weights['wR'] * (rj_cpu[k] + rj_gpu[k])

        # 3.8 Capacity slack penalty [was 3.7]
        obj += self.weights['wC'] * ca.sum1(slack_theta)

        # ===== CREATE NLP =====
        nlp = {
            'x': ca.vertcat(a_cpu, a_gpu, rj_cpu, rj_gpu, x_cpu, x_gpu, slack_theta),
            'f': obj,
            'g': ca.vertcat(*g) if g else ca.MX.sym('empty', 0),
            'p': ca.vertcat(p_theta, p_u, p_q, p_T_amb, p_T_target,
                          p_J_cpu_in, p_J_gpu_in,
                          p_r_bar_cpu, p_r_bar_gpu,
                          p_c_cpu, p_c_gpu)
        }

        # Solver options
        opts = {
            'ipopt.print_level': 0 if not self.verbose else 5,
            'ipopt.max_iter': 100,  # Increased for aggregate flows
            'ipopt.tol': 1e-4,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_strategy': 'adaptive',
            'print_time': 0 if not self.verbose else 1
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Store problem dimensions for later use
        self.n_cpu_clusters = n_cpu_clusters
        self.n_gpu_clusters = n_gpu_clusters

        # Store constraint bounds
        self.lbg = lbg
        self.ubg = ubg

        # Variable bounds
        n_vars = 4*N + (n_cpu_clusters + n_gpu_clusters)*N + N
        self.lbx = [0.0] * n_vars  # All variables >= 0
        self.ubx = [ca.inf] * n_vars  # No upper bound (constraints handle capacity)

        if self.verbose:
            print(f"[Stage2-DC{self.dc_id}] Aggregate-flow MPC problem built.")
            print(f"[Stage2-DC{self.dc_id}] Variables: {n_vars} (vs {self.max_jobs_per_dc * (M_dc + 1) + M_dc} before)")
            print(f"[Stage2-DC{self.dc_id}] Constraints: {len(lbg)}")

    def solve(self, jobs: List[Tuple[int, Dict]], state: Dict, T_target: float, enable_cold_start_diagnostics: bool = False) -> Dict[int, int]:
        """
        Solve aggregate-flow MPC for this DC.

        Args:
            jobs: List of (global_job_idx, job_dict) tuples for this DC
            state: Environment state (for cluster utilization, queues, temperature)
            T_target: Temperature setpoint from Stage 1
            enable_cold_start_diagnostics: Enable detailed cold-start logging

        Returns:
            Dict mapping global_job_idx → cluster_id (or env.M for rejection)
        """
        start_time = time.time()

        # NO TRUNCATION - can handle arbitrary number of jobs
        n_jobs = len(jobs)

        if n_jobs == 0:
            return {}

        # ===== 4.1 NEW INPUT PROCESSING =====
        # Separate jobs by type
        cpu_jobs = [(idx, job) for idx, job in jobs if job.get('tau', 'cpu') == 'cpu']
        gpu_jobs = [(idx, job) for idx, job in jobs if job.get('tau', 'cpu') == 'gpu']

        n_cpu_jobs = len(cpu_jobs)
        n_gpu_jobs = len(gpu_jobs)

        # Compute aggregate statistics
        r_bar_cpu = np.mean([job['r'] for _, job in cpu_jobs]) if n_cpu_jobs > 0 else 0.0
        r_bar_gpu = np.mean([job['r'] for _, job in gpu_jobs]) if n_gpu_jobs > 0 else 0.0

        # All jobs arrive at k=0 (current timestep)
        J_cpu_in = np.zeros(self.N)
        J_gpu_in = np.zeros(self.N)
        J_cpu_in[0] = n_cpu_jobs
        J_gpu_in[0] = n_gpu_jobs

        # ===== 4.2 NEW PARAMETER PACKING =====
        # Extract DC state
        dc_id = self.dc_id
        theta_dc = state.get('datacenter_temp', [22.0] * self.env.D)[dc_id]
        dc_params = self.env.config['datacenter_params'][dc_id]
        T_amb_dc = dc_params['climate']['base_temp']

        # Cluster utilizations and queues
        u = np.zeros(self.M_dc)
        q = np.zeros(self.M_dc)

        for idx, cluster_id in enumerate(self.cluster_ids):
            u[idx] = state.get('cluster_utilization', {}). get(cluster_id, 0.0)
            q[idx] = state.get('cluster_queue', {}).get(cluster_id, 0.0)

        # FIX A: Use RESIDUAL capacities (available headroom), not total capacities
        c_cpu = []
        c_gpu = []

        for idx, cluster_id in enumerate(self.cluster_ids):
            # CRITICAL BUG FIX: Use residual capacity (c - u), not total capacity
            # This is what's actually available for new jobs
            cluster = self.env.clusters[cluster_id]
            capacity = cluster.get('c', 10000.0)
            used = cluster.get('u', 0.0)
            residual = max(0.0, capacity - used)  # Available headroom

            if self.cluster_types[idx] == 'cpu':
                c_cpu.append(residual)
            else:
                c_gpu.append(residual)

        c_cpu = np.array(c_cpu) if c_cpu else np.array([0.0])
        c_gpu = np.array(c_gpu) if c_gpu else np.array([0.0])

        # ===== COLD START DIAGNOSTICS (ChatGPT recommendation) =====
        # Compute max feasible for diagnostics (even if not printing)
        H_cpu = np.sum(c_cpu)
        H_gpu = np.sum(c_gpu)
        A_cpu_max = int(np.floor(H_cpu / r_bar_cpu)) if r_bar_cpu > 0 else 0
        A_gpu_max = int(np.floor(H_gpu / r_bar_gpu)) if r_bar_gpu > 0 else 0

        if enable_cold_start_diagnostics:
            print(f"\n{'='*60}")
            print(f"COLD START DIAGNOSTICS - DC {self.dc_id}")
            print(f"{'='*60}")
            print(f"CPU:")
            print(f"  Jobs arriving: {n_cpu_jobs}")
            print(f"  Residual headroom (H_cpu): {H_cpu:.0f} CU")
            print(f"  Avg job size (r_bar_cpu): {r_bar_cpu:.0f} CU")
            print(f"  Max feasible (H/r_bar): {A_cpu_max} jobs")
            print(f"  Per-cluster headroom: {c_cpu}")
            print(f"GPU:")
            print(f"  Jobs arriving: {n_gpu_jobs}")
            print(f"  Residual headroom (H_gpu): {H_gpu:.0f} CU")
            print(f"  Avg job size (r_bar_gpu): {r_bar_gpu:.0f} CU")
            print(f"  Max feasible (H/r_bar): {A_gpu_max} jobs")
            print(f"  Per-cluster headroom: {c_gpu}")
            print(f"Thermal state:")
            print(f"  Current temp (theta): {theta_dc:.2f}°C")
            print(f"  Target temp (T_target): {T_target:.2f}°C")
            print(f"  Ambient temp (T_amb): {T_amb_dc:.2f}°C")
            print(f"MPC weights:")
            print(f"  wT (thermal): {self.weights['wT']}")
            print(f"  wE (energy): {self.weights['wE']}")
            print(f"  wR (rejection): {self.weights['wR']}")
            print(f"{'='*60}\n")

        # Pack parameters
        p_values = np.concatenate([
            [theta_dc],           # DC temperature
            u,                    # Cluster utilizations
            q,                    # Cluster queues
            [T_amb_dc],           # Ambient temperature
            [T_target],           # From Stage 1
            J_cpu_in,             # CPU job arrivals per timestep (N values)
            J_gpu_in,             # GPU job arrivals per timestep (N values)
            [r_bar_cpu],          # Average CPU job size
            [r_bar_gpu],          # Average GPU job size
            c_cpu,                # CPU cluster capacities
            c_gpu                 # GPU cluster capacities
        ])

        # Solve
        try:
            sol = self.solver(
                x0=self.x0,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p_values
            )

            if not self.solver.stats()['success']:
                # Fallback to greedy
                return self._greedy_fallback(jobs)

            # ===== 4.3 NEW SOLUTION EXTRACTION =====
            x_opt = sol['x'].full().flatten()

            # Parse decision variables
            offset = 0
            a_cpu_opt = x_opt[offset:offset+self.N]
            offset += self.N
            a_gpu_opt = x_opt[offset:offset+self.N]
            offset += self.N
            rj_cpu_opt = x_opt[offset:offset+self.N]
            offset += self.N
            rj_gpu_opt = x_opt[offset:offset+self.N]
            offset += self.N

            x_cpu_opt = x_opt[offset:offset+self.n_cpu_clusters*self.N].reshape(self.N, self.n_cpu_clusters)
            offset += self.n_cpu_clusters * self.N
            x_gpu_opt = x_opt[offset:offset+self.n_gpu_clusters*self.N].reshape(self.N, self.n_gpu_clusters)

            # ROUNDING: Convert continuous flows to discrete job assignments
            assignments = self._round_to_discrete_assignments(
                cpu_jobs, gpu_jobs,
                a_cpu_opt[0], a_gpu_opt[0],  # Use k=0 (current timestep)
                rj_cpu_opt[0], rj_gpu_opt[0],
                x_cpu_opt[0, :], x_gpu_opt[0, :]
            )

            # Update warm start
            self.x0 = x_opt

            solve_time = time.time() - start_time
            self.solve_times.append(solve_time)

            if self.verbose:
                print(f"[Stage2-DC{self.dc_id}] Solved {n_jobs} jobs in {solve_time*1000:.1f}ms")
                n_rejected = sum(1 for cid in assignments.values() if cid == self.env.M)
                print(f"[Stage2-DC{self.dc_id}] Admitted: CPU={int(a_cpu_opt[0])}/{n_cpu_jobs}, GPU={int(a_gpu_opt[0])}/{n_gpu_jobs}")
                print(f"[Stage2-DC{self.dc_id}] Rejected: {n_rejected}/{n_jobs}")

            # Cold start diagnostics continuation
            if enable_cold_start_diagnostics:
                print(f"MPC SOLUTION - DC {self.dc_id}:")
                print(f"  CPU admitted (continuous): {a_cpu_opt[0]:.2f} vs max_feasible {A_cpu_max}")
                print(f"  GPU admitted (continuous): {a_gpu_opt[0]:.2f} vs max_feasible {A_gpu_max}")
                print(f"  CPU rejected (continuous): {rj_cpu_opt[0]:.2f}")
                print(f"  GPU rejected (continuous): {rj_gpu_opt[0]:.2f}")
                print(f"  Utilization gap: CPU admits only {100*a_cpu_opt[0]/max(A_cpu_max,1):.1f}% of feasible")
                print(f"  Utilization gap: GPU admits only {100*a_gpu_opt[0]/max(A_gpu_max,1):.1f}% of feasible")
                print()

            # Return assignments with diagnostic info
            return {
                'assignments': assignments,
                'diagnostics': {
                    'a_cpu_planned': float(a_cpu_opt[0]),
                    'a_gpu_planned': float(a_gpu_opt[0]),
                    'rj_cpu_planned': float(rj_cpu_opt[0]),
                    'rj_gpu_planned': float(rj_gpu_opt[0]),
                    'n_cpu_jobs': n_cpu_jobs,
                    'n_gpu_jobs': n_gpu_jobs,
                    'n_assigned_cpu': sum(1 for idx, job in cpu_jobs if assignments.get(idx, self.env.M) != self.env.M),
                    'n_assigned_gpu': sum(1 for idx, job in gpu_jobs if assignments.get(idx, self.env.M) != self.env.M),
                    'n_rejected': sum(1 for cid in assignments.values() if cid == self.env.M)
                }
            }

        except Exception as e:
            if self.verbose:
                print(f"[Stage2-DC{self.dc_id}] Solver failed: {e}, using greedy fallback")

            # Greedy fallback
            assignments = self._greedy_fallback(jobs)
            return {
                'assignments': assignments,
                'diagnostics': {
                    'fallback': True,
                    'n_cpu_jobs': n_cpu_jobs,
                    'n_gpu_jobs': n_gpu_jobs,
                    'n_rejected': sum(1 for cid in assignments.values() if cid == self.env.M)
                }
            }

    def _round_to_discrete_assignments(
        self,
        cpu_jobs: List[Tuple[int, Dict]],
        gpu_jobs: List[Tuple[int, Dict]],
        a_cpu: float,
        a_gpu: float,
        rj_cpu: float,
        rj_gpu: float,
        x_cpu: np.ndarray,  # Shape: (n_cpu_clusters,)
        x_gpu: np.ndarray   # Shape: (n_gpu_clusters,)
    ) -> Dict[int, int]:
        """
        FIX C: Feasibility-preserving discrete job assignment.

        Ensures that:
        1. Per-cluster allocations never exceed residual capacity (in job slots)
        2. Redistributes overflow to clusters with headroom
        3. Rejects jobs that cannot fit anywhere

        Args:
            cpu_jobs, gpu_jobs: Lists of (global_idx, job_dict)
            a_cpu, a_gpu: Continuous admission counts from MPC
            rj_cpu, rj_gpu: Continuous rejection counts
            x_cpu, x_gpu: Continuous cluster flow allocations

        Returns:
            Dict mapping global_job_idx → cluster_id (or env.M for rejection)
        """
        assignments = {}

        # Build cluster lists by type (global IDs)
        cpu_cluster_ids = [self.cluster_ids[i] for i in range(len(self.cluster_ids))
                           if self.cluster_types[i] == 'cpu']
        gpu_cluster_ids = [self.cluster_ids[i] for i in range(len(self.cluster_ids))
                           if self.cluster_types[i] == 'gpu']

        # Helper: get residual capacity in resource units for a cluster
        def residual_cu(cluster_global_id: int) -> float:
            cl = self.env.clusters[cluster_global_id]
            return max(0.0, float(cl.get('c', 0.0) - cl.get('u', 0.0)))

        # ----- CPU Jobs -----
        n_cpu_jobs = len(cpu_jobs)
        if n_cpu_jobs > 0 and len(cpu_cluster_ids) > 0:
            # Use average job size (same as MPC optimization)
            r_bar_cpu = float(np.mean([job['r'] for _, job in cpu_jobs])) if n_cpu_jobs > 0 else 0.0
            r_bar_cpu = max(r_bar_cpu, 1e-6)

            # Compute per-cluster max feasible jobs from residual capacity
            cpu_max_jobs = np.array([int(np.floor(residual_cu(cid) / r_bar_cpu)) for cid in cpu_cluster_ids], dtype=int)
            total_cpu_feasible = int(np.sum(cpu_max_jobs))

            # Cannot admit more than arrivals or feasible capacity
            n_cpu_admit = int(np.round(a_cpu))
            n_cpu_admit = max(0, min(n_cpu_admit, n_cpu_jobs, total_cpu_feasible))

            if n_cpu_admit > 0:
                # Target allocations from MPC proportions
                xsum = float(np.sum(x_cpu))
                if xsum > 1e-9:
                    frac = x_cpu / xsum
                else:
                    frac = np.ones(len(cpu_cluster_ids)) / len(cpu_cluster_ids)

                desired = frac * n_cpu_admit
                alloc = np.floor(desired).astype(int)
                rem = n_cpu_admit - int(np.sum(alloc))

                # Largest remainder method
                if rem > 0:
                    fracs = desired - alloc
                    for j in np.argsort(-fracs):
                        if rem == 0:
                            break
                        alloc[j] += 1
                        rem -= 1

                # CRITICAL: Enforce feasibility by clipping to cpu_max_jobs
                alloc = np.minimum(alloc, cpu_max_jobs)
                to_place = n_cpu_admit - int(np.sum(alloc))

                # Redistribute overflow to clusters with headroom
                if to_place > 0:
                    headroom = cpu_max_jobs - alloc
                    order = np.argsort(-headroom)  # Fill largest headroom first
                    for j in order:
                        if to_place == 0:
                            break
                        add = min(headroom[j], to_place)
                        if add > 0:
                            alloc[j] += add
                            to_place -= add

                # Assign jobs to clusters
                jidx = 0
                for local_i, n_alloc in enumerate(alloc):
                    cid = cpu_cluster_ids[local_i]
                    for _ in range(int(n_alloc)):
                        if jidx < len(cpu_jobs):
                            global_job_idx, _ = cpu_jobs[jidx]
                            assignments[global_job_idx] = cid
                            jidx += 1

                # Reject remaining jobs
                for i in range(jidx, n_cpu_jobs):
                    global_job_idx, _ = cpu_jobs[i]
                    assignments[global_job_idx] = self.env.M
            else:
                # Reject all CPU jobs
                for global_job_idx, _ in cpu_jobs:
                    assignments[global_job_idx] = self.env.M
        else:
            # No CPU clusters or jobs
            for global_job_idx, _ in cpu_jobs:
                assignments[global_job_idx] = self.env.M

        # ----- GPU Jobs -----
        n_gpu_jobs = len(gpu_jobs)
        if n_gpu_jobs > 0 and len(gpu_cluster_ids) > 0:
            r_bar_gpu = float(np.mean([job['r'] for _, job in gpu_jobs])) if n_gpu_jobs > 0 else 0.0
            r_bar_gpu = max(r_bar_gpu, 1e-6)

            gpu_max_jobs = np.array([int(np.floor(residual_cu(cid) / r_bar_gpu)) for cid in gpu_cluster_ids], dtype=int)
            total_gpu_feasible = int(np.sum(gpu_max_jobs))

            n_gpu_admit = int(np.round(a_gpu))
            n_gpu_admit = max(0, min(n_gpu_admit, n_gpu_jobs, total_gpu_feasible))

            if n_gpu_admit > 0:
                xsum = float(np.sum(x_gpu))
                if xsum > 1e-9:
                    frac = x_gpu / xsum
                else:
                    frac = np.ones(len(gpu_cluster_ids)) / len(gpu_cluster_ids)

                desired = frac * n_gpu_admit
                alloc = np.floor(desired).astype(int)
                rem = n_gpu_admit - int(np.sum(alloc))

                if rem > 0:
                    fracs = desired - alloc
                    for j in np.argsort(-fracs):
                        if rem == 0:
                            break
                        alloc[j] += 1
                        rem -= 1

                alloc = np.minimum(alloc, gpu_max_jobs)
                to_place = n_gpu_admit - int(np.sum(alloc))

                if to_place > 0:
                    headroom = gpu_max_jobs - alloc
                    order = np.argsort(-headroom)
                    for j in order:
                        if to_place == 0:
                            break
                        add = min(headroom[j], to_place)
                        if add > 0:
                            alloc[j] += add
                            to_place -= add

                jidx = 0
                for local_i, n_alloc in enumerate(alloc):
                    cid = gpu_cluster_ids[local_i]
                    for _ in range(int(n_alloc)):
                        if jidx < len(gpu_jobs):
                            global_job_idx, _ = gpu_jobs[jidx]
                            assignments[global_job_idx] = cid
                            jidx += 1

                for i in range(jidx, n_gpu_jobs):
                    global_job_idx, _ = gpu_jobs[i]
                    assignments[global_job_idx] = self.env.M
            else:
                for global_job_idx, _ in gpu_jobs:
                    assignments[global_job_idx] = self.env.M
        else:
            for global_job_idx, _ in gpu_jobs:
                assignments[global_job_idx] = self.env.M

        return assignments

    def _proportional_round(self, x: np.ndarray) -> np.ndarray:
        """
        Proportional rounding with largest remainder method.
        Ensures sum(rounded) = round(sum(x)).

        Example: [2.3, 1.7, 0.9] → [2, 2, 1] (sum=5)
        """
        floored = np.floor(x).astype(int)
        remainder_total = int(np.round(np.sum(x) - np.sum(floored)))

        if remainder_total > 0:
            fracs = x - floored
            top_indices = np.argsort(-fracs)[:remainder_total]
            floored[top_indices] += 1

        return floored

    def _greedy_fallback(self, jobs: List[Tuple[int, Dict]]) -> Dict[int, int]:
        """Greedy fallback if solver fails."""
        assignments = {}

        for global_job_idx, job in jobs:
            best_cluster = self.env.M  # Default: reject
            best_ratio = float('inf')

            for cluster_id in self.cluster_ids:
                cluster = self.env.clusters[cluster_id]
                cluster_params = self.env.config['cluster_params'][cluster_id]

                # Check affinity
                if job.get('tau', 'cpu') != cluster_params.get('type', 'cpu'):
                    continue

                available = cluster.get('c', 0) - cluster.get('u', 0)

                if available >= job['r']:
                    capacity = cluster.get('c', 1)
                    util_ratio = cluster.get('u', 0) / capacity if capacity > 0 else 0

                    if util_ratio < best_ratio:
                        best_ratio = util_ratio
                        best_cluster = cluster_id

            assignments[global_job_idx] = best_cluster

        return assignments


class HierarchicalMPCScheduler(SchedulingPolicy):
    """
    Main Hierarchical MPC Scheduler.

    Coordinates two-stage optimization:
    1. Stage 1: DC-level ratios and temperature setpoints
    2. Stage 2: Per-DC cluster-level job assignments
    """

    def __init__(self, env, N1: int = 6, N2: int = 3,
                 weights_s1: Optional[Dict] = None,
                 weights_s2: Optional[Dict] = None,
                 verbose: bool = False,
                 enable_diagnostics: bool = False):
        """
        Initialize Hierarchical MPC Scheduler with aggregate-flow Stage 2.

        Args:
            env: DataCenterEnv instance
            N1: Stage 1 horizon (6 = 30 min)
            N2: Stage 2 horizon (3 = 15 min)
            weights_s1: Stage 1 weights
            weights_s2: Stage 2 weights (includes wE, wSmooth)
            verbose: Print debug information
            enable_diagnostics: Enable detailed per-timestep logging
        """
        super().__init__(env)
        self.N1 = N1
        self.N2 = N2
        self.verbose = verbose
        self.enable_diagnostics = enable_diagnostics

        # Build Stage 1 MPC
        print(f"[HierarchicalMPC] Building Stage 1 (DC-level MPC)...")
        self.stage1 = HierarchicalMPCStage1(env, N=N1, weights=weights_s1, verbose=verbose)
        print(f"[HierarchicalMPC] Stage 1 built successfully!")

        # Build Stage 2 MPCs (one per DC) - now with aggregate flows
        print(f"[HierarchicalMPC] Building Stage 2 (per-DC aggregate-flow MPCs)...")
        self.stage2_solvers = {}
        for dc_id in range(env.D):
            self.stage2_solvers[dc_id] = HierarchicalMPCStage2(
                env, dc_id, N=N2,
                weights=weights_s2, verbose=verbose
            )
        print(f"[HierarchicalMPC] Stage 2 built successfully!")

        # Performance tracking
        self.solve_times = {'stage1': [], 'stage2': [], 'total': []}

        # Diagnostic logging
        self.diagnostics = [] if enable_diagnostics else None
        self.timestep = 0

        if verbose:
            print(f"[HierarchicalMPC] Initialized with N1={N1}, N2={N2}")
            if enable_diagnostics:
                print(f"[HierarchicalMPC] Diagnostic logging ENABLED")

    def select_action(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Select action using two-stage hierarchical MPC.

        Returns:
            Dict with job_assignment array and T_target array
        """
        start_time = time.time()

        # Get current state and jobs
        state = self._extract_state()
        jobs = self.env.current_jobs
        n_jobs = len(jobs)

        if n_jobs == 0:
            return {
                "job_assignment": np.array([], dtype=int),
                "T_target": np.full(self.env.D, 22.0, dtype=np.float32)
            }

        # Initialize diagnostic record
        if self.enable_diagnostics:
            diag = {
                'timestep': self.timestep,
                'n_jobs': n_jobs,
                'n_cpu_arrived': sum(1 for j in jobs if j.get('tau', 'cpu') == 'cpu'),
                'n_gpu_arrived': sum(1 for j in jobs if j.get('tau', 'cpu') == 'gpu'),
                'stage1_quotas': {},
                'stage2_plans': {},
                'dispatch_realized': {},
                'cluster_headroom': {}
            }

        # STAGE 1: Solve for DC ratios and T_target
        stage1_start = time.time()
        stage1_result = self.stage1.solve(state, jobs)
        stage1_time = time.time() - stage1_start
        self.solve_times['stage1'].append(stage1_time)

        if self.verbose:
            print(f"[HierarchicalMPC] Stage 1 completed in {stage1_time*1000:.1f}ms")

        # Capture Stage 1 quotas
        if self.enable_diagnostics:
            diag['stage1_quotas'] = {
                'cpu_per_dc': stage1_result['job_counts']['cpu'].tolist(),
                'gpu_per_dc': stage1_result['job_counts']['gpu'].tolist(),
                'total_cpu_quota': int(np.sum(stage1_result['job_counts']['cpu'])),
                'total_gpu_quota': int(np.sum(stage1_result['job_counts']['gpu']))
            }

        # Distribute jobs to DCs per Stage 1 allocation
        jobs_per_dc = self._distribute_jobs_to_dcs(jobs, stage1_result['job_counts'])

        # Capture cluster headroom (feasibility snapshot)
        if self.enable_diagnostics:
            for i in range(self.env.M):
                cluster = self.env.clusters[i]
                headroom = max(0.0, cluster.get('c', 0) - cluster.get('u', 0))
                diag['cluster_headroom'][i] = {
                    'headroom': float(headroom),
                    'utilization': float(cluster.get('u', 0) / cluster.get('c', 1)),
                    'type': self.env.config['cluster_params'][i].get('type', 'cpu')
                }

        # STAGE 2: Solve per-DC cluster assignment (sequential)
        stage2_start = time.time()
        job_assignments_per_dc = {}
        stage2_diagnostics = {}

        # Enable cold-start diagnostics for timestep 0
        enable_cold_start = self.enable_diagnostics and self.timestep == 0

        for dc_id in range(self.env.D):
            if len(jobs_per_dc[dc_id]) > 0:
                result = self.stage2_solvers[dc_id].solve(
                    jobs_per_dc[dc_id],
                    state,
                    stage1_result['T_target'][dc_id],
                    enable_cold_start_diagnostics=enable_cold_start
                )
                job_assignments_per_dc[dc_id] = result['assignments']
                stage2_diagnostics[dc_id] = result.get('diagnostics', {})

        stage2_time = time.time() - stage2_start
        self.solve_times['stage2'].append(stage2_time)

        if self.verbose:
            print(f"[HierarchicalMPC] Stage 2 completed in {stage2_time*1000:.1f}ms")

        # Capture Stage 2 plans
        if self.enable_diagnostics:
            diag['stage2_plans'] = stage2_diagnostics

        # Reconstruct final assignment array
        job_assignment = self._reconstruct_assignment(jobs_per_dc, job_assignments_per_dc)

        # Capture dispatch realized
        if self.enable_diagnostics:
            n_assigned = sum(1 for cid in job_assignment if cid != self.env.M)
            n_rejected = sum(1 for cid in job_assignment if cid == self.env.M)
            n_assigned_cpu = sum(1 for j_idx, cid in enumerate(job_assignment)
                                 if cid != self.env.M and jobs[j_idx].get('tau', 'cpu') == 'cpu')
            n_assigned_gpu = sum(1 for j_idx, cid in enumerate(job_assignment)
                                 if cid != self.env.M and jobs[j_idx].get('tau', 'cpu') == 'gpu')

            diag['dispatch_realized'] = {
                'n_assigned': n_assigned,
                'n_rejected': n_rejected,
                'n_assigned_cpu': n_assigned_cpu,
                'n_assigned_gpu': n_assigned_gpu,
                'rejection_rate': n_rejected / n_jobs if n_jobs > 0 else 0
            }

            # Append to diagnostics log
            self.diagnostics.append(diag)

        total_time = time.time() - start_time
        self.solve_times['total'].append(total_time)

        if self.verbose:
            print(f"[HierarchicalMPC] Total solve time: {total_time*1000:.1f}ms")
            print(f"[HierarchicalMPC] Assigned {n_jobs} jobs")

        # Increment timestep for diagnostics
        if self.enable_diagnostics:
            self.timestep += 1

        return {
            "job_assignment": job_assignment,
            "T_target": stage1_result['T_target'].astype(np.float32)
        }

    def _extract_state(self) -> Dict[str, np.ndarray]:
        """Extract current state aggregated by DC."""
        q_cpu = np.zeros(self.env.D)
        q_gpu = np.zeros(self.env.D)
        u_cpu = np.zeros(self.env.D)
        u_gpu = np.zeros(self.env.D)

        for i in range(self.env.M):
            dc_id = self.env.cluster_to_dc[i]
            cluster = self.env.clusters[i]
            cluster_params = self.env.config['cluster_params'][i]

            if cluster_params.get('type', 'cpu') == 'cpu':
                q_cpu[dc_id] += cluster.get('q', 0)
                u_cpu[dc_id] += cluster.get('u', 0)
            else:
                q_gpu[dc_id] += cluster.get('q', 0)
                u_gpu[dc_id] += cluster.get('u', 0)

        theta = np.array([self.env.datacenters[d].get('theta', 22.0) for d in range(self.env.D)])
        T_amb = np.array([self.env.datacenters[d].get('T_amb', 15.0) for d in range(self.env.D)])

        return {
            'theta': theta,
            'q_cpu': q_cpu,
            'q_gpu': q_gpu,
            'u_cpu': u_cpu,
            'u_gpu': u_gpu,
            'T_amb': T_amb
        }

    def _distribute_jobs_to_dcs(self, jobs: List[Dict],
                                job_counts: Dict[str, np.ndarray]) -> Dict[int, List[Tuple[int, Dict]]]:
        """
        Distribute jobs to DCs based on Stage 1 allocation.

        Returns:
            Dict mapping dc_id -> List of (global_job_idx, job_dict)
        """
        jobs_per_dc = {dc_id: [] for dc_id in range(self.env.D)}

        # Separate jobs by type
        cpu_jobs = [(idx, j) for idx, j in enumerate(jobs) if j.get('tau', 'cpu') == 'cpu']
        gpu_jobs = [(idx, j) for idx, j in enumerate(jobs) if j.get('tau', 'cpu') == 'gpu']

        # Distribute CPU jobs
        cpu_counts = job_counts['cpu']
        cpu_job_idx = 0
        for dc_id in range(self.env.D):
            n_jobs_dc = int(cpu_counts[dc_id])
            for _ in range(n_jobs_dc):
                if cpu_job_idx < len(cpu_jobs):
                    jobs_per_dc[dc_id].append(cpu_jobs[cpu_job_idx])
                    cpu_job_idx += 1

        # Distribute GPU jobs
        gpu_counts = job_counts['gpu']
        gpu_job_idx = 0
        for dc_id in range(self.env.D):
            n_jobs_dc = int(gpu_counts[dc_id])
            for _ in range(n_jobs_dc):
                if gpu_job_idx < len(gpu_jobs):
                    jobs_per_dc[dc_id].append(gpu_jobs[gpu_job_idx])
                    gpu_job_idx += 1

        return jobs_per_dc

    def _reconstruct_assignment(self, jobs_per_dc: Dict[int, List[Tuple[int, Dict]]],
                                job_assignments_per_dc: Dict[int, Dict[int, int]]) -> np.ndarray:
        """
        Reconstruct global job assignment array from per-DC results.

        Returns:
            np.ndarray of cluster IDs (or M for reject)
        """
        n_total_jobs = len(self.env.current_jobs)
        job_assignment = np.full(n_total_jobs, self.env.M, dtype=int)  # Default: reject

        # Fill in assignments from each DC
        for dc_id in range(self.env.D):
            if dc_id in job_assignments_per_dc:
                assignments = job_assignments_per_dc[dc_id]
                for global_job_idx, cluster_id in assignments.items():
                    job_assignment[global_job_idx] = cluster_id

        return job_assignment

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        stats = {}

        for stage in ['stage1', 'stage2', 'total']:
            if self.solve_times[stage]:
                times = np.array(self.solve_times[stage]) * 1000  # Convert to ms
                stats[f'{stage}_avg_ms'] = np.mean(times)
                stats[f'{stage}_max_ms'] = np.max(times)
                stats[f'{stage}_min_ms'] = np.min(times)
                stats[f'{stage}_std_ms'] = np.std(times)

        return stats

    def get_diagnostics(self) -> List[Dict]:
        """Get diagnostic logs (if enabled)."""
        if self.diagnostics is None:
            return []
        return self.diagnostics

    def export_diagnostics(self, filepath: str):
        """Export diagnostics to JSON file."""
        import json
        if self.diagnostics is None:
            print("[HierarchicalMPC] Diagnostics not enabled, nothing to export")
            return

        with open(filepath, 'w') as f:
            json.dump(self.diagnostics, f, indent=2)
        print(f"[HierarchicalMPC] Diagnostics exported to {filepath}")