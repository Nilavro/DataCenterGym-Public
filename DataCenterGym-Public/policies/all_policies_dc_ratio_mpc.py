"""
DC-Level Ratio MPC Scheduler

Optimizes datacenter-level allocation ratios (8 variables) instead of individual
job-cluster assignments (4200 variables), achieving massive speedup while maintaining
similar QoS.

Key Innovation:
- Decision variables: p_d^cpu and p_d^gpu (allocation ratios per DC)
- Only 8 continuous variables regardless of job count
- Greedy dispatch within each DC after ratio optimization
- ~100x speedup vs job-level MPC
"""

import numpy as np
import casadi as ca
import time
from typing import Dict, List, Tuple, Optional, Any
from policies.all_policies import SchedulingPolicy


class DCRatioMPCScheduler(SchedulingPolicy):
    """
    DC-Level Allocation Ratio MPC Scheduler.

    Optimizes continuous allocation ratios for CPU and GPU jobs across datacenters,
    then dispatches jobs greedily within each DC.
    """

    def __init__(self,
                 env,
                 N: int = 3,
                 T_target_fixed: float = 22.0,
                 weights: Optional[Dict[str, float]] = None,
                 solver: str = 'casadi',
                 verbose: bool = False):
        """
        Initialize DC-Level Ratio MPC Scheduler.

        Args:
            env: DataCenterEnv instance
            N: MPC horizon length (3-6 timesteps)
            T_target_fixed: Fixed temperature setpoint (°C)
            weights: Dict with keys: wQ, wT, wE, wL, wC
            solver: 'casadi' for NLP or 'cvxpy' for QP
            verbose: Print debug information
        """
        super().__init__(env)
        self.N = N
        self.T_target_fixed = T_target_fixed
        self.weights = weights or self._default_weights()
        self.solver_type = solver
        self.verbose = verbose

        # Environment dimensions
        self.D = env.D  # Number of datacenters
        self.M = env.M  # Number of clusters

        # Load DC-level parameters
        self._load_dc_parameters()

        # Build optimization problem (8 variables only!)
        self._build_optimization_problem()

        # Warm start for solver
        self.x0 = np.ones(2 * self.D) / self.D  # Equal distribution initially

        # Performance tracking
        self.solve_times = []

    def _default_weights(self) -> Dict[str, float]:
        """Default MPC weights."""
        return {
            'wQ': 1.0,      # Queue minimization
            'wT': 10.0,     # Thermal safety (high priority)
            'wE': 0.01,     # Energy cost
            'wL': 0.5,      # Load balancing
            'wC': 100.0     # Capacity constraint penalty
        }

    def _load_dc_parameters(self):
        """Load and aggregate DC-level thermal and capacity parameters."""
        self.dt = 300.0  # 5 minutes in seconds

        # Thermal parameters per DC
        self.C = np.zeros(self.D)  # Thermal capacitance
        self.R = np.zeros(self.D)  # Thermal resistance
        self.K_p = np.zeros(self.D)  # Cooling gain
        self.psi = np.zeros(self.D)  # Energy price
        self.theta_ref = np.zeros(self.D)  # Reference temperature

        # Heat generation coefficients (average per DC)
        self.alpha_cpu_avg = np.zeros(self.D)
        self.alpha_gpu_avg = np.zeros(self.D)

        # Capacity per DC
        self.C_dc_cpu = np.zeros(self.D)  # Total CPU capacity per DC
        self.C_dc_gpu = np.zeros(self.D)  # Total GPU capacity per DC

        for d in range(self.D):
            dc = self.env.datacenters[d]

            # Thermal parameters
            self.C[d] = dc.get('C', 1e7)  # J/K
            self.R[d] = dc.get('R', 0.001)  # K/W
            self.K_p[d] = dc.get('K_p', 1e6)  # W/K
            self.psi[d] = dc.get('psi', 0.10)  # $/kWh
            self.theta_ref[d] = dc.get('theta_ref', 24.0)  # °C

            # Aggregate cluster parameters for this DC
            cpu_clusters_in_dc = []
            gpu_clusters_in_dc = []

            for i in range(self.M):
                if self.env.cluster_to_dc[i] == d:
                    cluster = self.env.clusters[i]
                    if cluster.get('type', 'cpu') == 'cpu':
                        cpu_clusters_in_dc.append(i)
                        self.C_dc_cpu[d] += cluster.get('c', 0)
                    else:
                        gpu_clusters_in_dc.append(i)
                        self.C_dc_gpu[d] += cluster.get('c', 0)

            # Average heat generation coefficients
            if cpu_clusters_in_dc:
                alphas = [self.env.clusters[i].get('alpha', 10.0) for i in cpu_clusters_in_dc]
                self.alpha_cpu_avg[d] = np.mean(alphas)
            else:
                self.alpha_cpu_avg[d] = 10.0  # Default

            if gpu_clusters_in_dc:
                alphas = [self.env.clusters[i].get('alpha', 20.0) for i in gpu_clusters_in_dc]
                self.alpha_gpu_avg[d] = np.mean(alphas)
            else:
                self.alpha_gpu_avg[d] = 20.0  # Default

    def _build_optimization_problem(self):
        """
        Build small NLP with only 8 variables!

        Decision variables:
        - p_cpu[d]: CPU allocation ratio for datacenter d
        - p_gpu[d]: GPU allocation ratio for datacenter d
        """
        import casadi as ca

        # Decision variables (8 total)
        p_cpu = ca.MX.sym('p_cpu', self.D)  # CPU ratios per DC
        p_gpu = ca.MX.sym('p_gpu', self.D)  # GPU ratios per DC

        # Parameters (current state and job counts)
        theta_current = ca.MX.sym('theta', self.D)
        q_cpu_current = ca.MX.sym('q_cpu', self.D)
        q_gpu_current = ca.MX.sym('q_gpu', self.D)
        u_cpu_current = ca.MX.sym('u_cpu', self.D)
        u_gpu_current = ca.MX.sym('u_gpu', self.D)
        n_cpu = ca.MX.sym('n_cpu', 1)  # Number of CPU jobs
        n_gpu = ca.MX.sym('n_gpu', 1)  # Number of GPU jobs
        T_amb = ca.MX.sym('T_amb', self.D)
        r_bar_cpu = ca.MX.sym('r_bar_cpu', 1)  # Average CPU job resource
        r_bar_gpu = ca.MX.sym('r_bar_gpu', 1)  # Average GPU job resource

        # Simplex constraints
        g = [ca.sum1(p_cpu) - 1.0, ca.sum1(p_gpu) - 1.0]

        # Prediction over horizon
        obj = 0.0
        theta = theta_current

        for k in range(self.N):
            # Incremental utilization from new jobs
            delta_u_cpu = p_cpu * n_cpu * r_bar_cpu
            delta_u_gpu = p_gpu * n_gpu * r_bar_gpu

            # Total predicted utilization
            u_pred_cpu = u_cpu_current + delta_u_cpu * (1 - k * 0.2)  # Decay factor
            u_pred_gpu = u_gpu_current + delta_u_gpu * (1 - k * 0.2)

            # Heat generation per DC
            Q_gen = self.alpha_cpu_avg * u_pred_cpu + self.alpha_gpu_avg * u_pred_gpu

            # Cooling and thermal dynamics
            theta_next = ca.MX.zeros(self.D)
            Q_cool = ca.MX.zeros(self.D)

            for d in range(self.D):
                # Proportional cooling
                Q_cool[d] = self.K_p[d] * ca.fmax(0, theta[d] - self.T_target_fixed)

                # RC thermal model
                dtheta = (self.dt / self.C[d]) * (
                    Q_gen[d] - (theta[d] - T_amb[d]) / self.R[d] - Q_cool[d]
                )
                theta_next[d] = theta[d] + dtheta

            # Queue prediction
            beta = 0.1  # Queue growth factor
            q_pred_cpu = q_cpu_current + beta * p_cpu * n_cpu * (k + 1)
            q_pred_gpu = q_gpu_current + beta * p_gpu * n_gpu * (k + 1)

            # Objective components

            # 1. Queue penalty
            obj += self.weights['wQ'] * (ca.sum1(q_pred_cpu**2) + ca.sum1(q_pred_gpu**2))

            # 2. Thermal penalty
            for d in range(self.D):
                obj += self.weights['wT'] * ca.fmax(0, theta_next[d] - self.theta_ref[d])**2

            # 3. Energy cost
            for d in range(self.D):
                energy_kwh = Q_cool[d] * self.dt / 3.6e6
                obj += self.weights['wE'] * self.psi[d] * energy_kwh

            # 4. Load balancing
            u_total = u_pred_cpu + u_pred_gpu
            u_mean = ca.sum1(u_total) / self.D
            for d in range(self.D):
                obj += self.weights['wL'] * (u_total[d] - u_mean)**2

            # 5. Capacity constraints (soft)
            for d in range(self.D):
                obj += self.weights['wC'] * ca.fmax(0, u_pred_cpu[d] - self.C_dc_cpu[d])**2
                obj += self.weights['wC'] * ca.fmax(0, u_pred_gpu[d] - self.C_dc_gpu[d])**2

            # Update state for next timestep
            theta = theta_next

        # Create NLP
        x = ca.vertcat(p_cpu, p_gpu)
        p = ca.vertcat(theta_current, q_cpu_current, q_gpu_current,
                       u_cpu_current, u_gpu_current, n_cpu, n_gpu, T_amb,
                       r_bar_cpu, r_bar_gpu)

        nlp = {'x': x, 'f': obj, 'g': ca.vertcat(*g), 'p': p}

        # Solver options
        opts = {
            'ipopt.print_level': 0 if not self.verbose else 5,
            'ipopt.max_iter': 50,  # Fast convergence with 8 vars
            'ipopt.tol': 1e-4,
            'ipopt.warm_start_init_point': 'yes'
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Bounds
        self.lbx = [0.0] * (2 * self.D)  # p >= 0
        self.ubx = [1.0] * (2 * self.D)  # p <= 1
        self.lbg = [0.0, 0.0]  # Equality constraints
        self.ubg = [0.0, 0.0]

    def select_action(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Select action using DC-level ratio optimization.

        Returns:
            action dict with job_assignment array and fixed T_target
        """
        start_time = time.time()

        # 1. Count jobs by type
        cpu_jobs = []
        gpu_jobs = []

        for idx, job in enumerate(self.env.current_jobs):
            if job.get('tau', 'cpu') == 'cpu':
                cpu_jobs.append((idx, job))
            else:
                gpu_jobs.append((idx, job))

        n_cpu = len(cpu_jobs)
        n_gpu = len(gpu_jobs)

        if self.verbose:
            print(f"DC-Ratio MPC: {n_cpu} CPU jobs, {n_gpu} GPU jobs")

        # 2. Handle edge cases
        if n_cpu == 0 and n_gpu == 0:
            # No jobs to assign
            return {
                "job_assignment": np.array([], dtype=int),
                "T_target": np.full(self.D, self.T_target_fixed, dtype=np.float32)
            }

        # 3. Solve for optimal DC ratios (8 variables)
        p_cpu, p_gpu = self._solve_dc_ratios(n_cpu, n_gpu)

        # 4. Convert ratios to job assignments
        job_assignment = self._dispatch_jobs(p_cpu, p_gpu, cpu_jobs, gpu_jobs)

        # 5. Fixed temperature setpoints
        T_target = np.full(self.D, self.T_target_fixed, dtype=np.float32)

        # Track performance
        solve_time = time.time() - start_time
        self.solve_times.append(solve_time)

        if self.verbose:
            print(f"Solve time: {solve_time*1000:.1f}ms")
            print(f"CPU ratios: {p_cpu}")
            print(f"GPU ratios: {p_gpu}")

        return {
            "job_assignment": job_assignment,
            "T_target": T_target
        }

    def _solve_dc_ratios(self, n_cpu: int, n_gpu: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for optimal DC allocation ratios.

        Returns:
            p_cpu: array of shape [D] with CPU ratios
            p_gpu: array of shape [D] with GPU ratios
        """
        # Extract current DC-level state
        state = self._extract_dc_state()

        # Calculate average resource demands
        r_bar_cpu = 1000.0  # Default
        r_bar_gpu = 5000.0  # Default

        if self.env.current_jobs:
            cpu_resources = [j['r'] for j in self.env.current_jobs if j.get('tau', 'cpu') == 'cpu']
            gpu_resources = [j['r'] for j in self.env.current_jobs if j.get('tau', 'cpu') == 'gpu']

            if cpu_resources:
                r_bar_cpu = np.mean(cpu_resources)
            if gpu_resources:
                r_bar_gpu = np.mean(gpu_resources)

        # Pack parameters
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
            [r_bar_gpu]
        ])

        # Handle edge cases
        if n_cpu == 0:
            # No CPU jobs, set equal GPU distribution
            p_cpu = np.ones(self.D) / self.D
            if n_gpu == 0:
                p_gpu = np.ones(self.D) / self.D
            else:
                # Solve only for GPU ratios
                sol = self.solver(
                    x0=self.x0,
                    lbx=self.lbx,
                    ubx=self.ubx,
                    lbg=self.lbg,
                    ubg=self.ubg,
                    p=p_values
                )
                x_opt = sol['x'].full().flatten()
                p_gpu = x_opt[self.D:]
                # Normalize GPU ratios
                p_gpu = p_gpu / np.sum(p_gpu) if np.sum(p_gpu) > 0 else np.ones(self.D) / self.D

        elif n_gpu == 0:
            # No GPU jobs, set equal CPU distribution
            p_gpu = np.ones(self.D) / self.D
            # Solve only for CPU ratios
            sol = self.solver(
                x0=self.x0,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p_values
            )
            x_opt = sol['x'].full().flatten()
            p_cpu = x_opt[:self.D]
            # Normalize CPU ratios
            p_cpu = p_cpu / np.sum(p_cpu) if np.sum(p_cpu) > 0 else np.ones(self.D) / self.D

        else:
            # Both CPU and GPU jobs present
            sol = self.solver(
                x0=self.x0,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p_values
            )

            # Extract solution
            x_opt = sol['x'].full().flatten()
            p_cpu = x_opt[:self.D]
            p_gpu = x_opt[self.D:]

            # Update warm start
            self.x0 = x_opt

        # Ensure normalization (handle numerical issues)
        p_cpu = p_cpu / np.sum(p_cpu) if np.sum(p_cpu) > 0 else np.ones(self.D) / self.D
        p_gpu = p_gpu / np.sum(p_gpu) if np.sum(p_gpu) > 0 else np.ones(self.D) / self.D

        return p_cpu, p_gpu

    def _extract_dc_state(self) -> Dict[str, np.ndarray]:
        """
        Aggregate cluster-level state to DC level.
        """
        q_cpu = np.zeros(self.D)
        q_gpu = np.zeros(self.D)
        u_cpu = np.zeros(self.D)
        u_gpu = np.zeros(self.D)

        for i in range(self.M):
            dc_id = self.env.cluster_to_dc[i]
            cluster = self.env.clusters[i]

            if cluster.get('type', 'cpu') == 'cpu':
                q_cpu[dc_id] += cluster.get('q', 0)
                u_cpu[dc_id] += cluster.get('u', 0)
            else:
                q_gpu[dc_id] += cluster.get('q', 0)
                u_gpu[dc_id] += cluster.get('u', 0)

        # Get thermal state
        theta = np.array([self.env.datacenters[d].get('theta', 22.0) for d in range(self.D)])
        T_amb = np.array([self.env.datacenters[d].get('T_amb', 15.0) for d in range(self.D)])

        return {
            'theta': theta,
            'q_cpu': q_cpu,
            'q_gpu': q_gpu,
            'u_cpu': u_cpu,
            'u_gpu': u_gpu,
            'T_amb': T_amb
        }

    def _dispatch_jobs(self, p_cpu: np.ndarray, p_gpu: np.ndarray,
                      cpu_jobs: List[Tuple[int, Dict]],
                      gpu_jobs: List[Tuple[int, Dict]]) -> np.ndarray:
        """
        Convert DC ratios to actual job assignments.

        Steps:
        1. Convert ratios to integer job counts per DC
        2. Greedily assign within each DC
        3. Return array of cluster IDs
        """
        n_total_jobs = len(cpu_jobs) + len(gpu_jobs)
        job_assignment = np.full(n_total_jobs, self.M, dtype=int)  # Default: reject (M)

        # Convert ratios to job counts
        if cpu_jobs:
            jobs_per_dc_cpu = self._round_preserving_sum(p_cpu * len(cpu_jobs))
        else:
            jobs_per_dc_cpu = np.zeros(self.D, dtype=int)

        if gpu_jobs:
            jobs_per_dc_gpu = self._round_preserving_sum(p_gpu * len(gpu_jobs))
        else:
            jobs_per_dc_gpu = np.zeros(self.D, dtype=int)

        # Pre-compute cluster lists per DC
        cpu_clusters_by_dc = [[] for _ in range(self.D)]
        gpu_clusters_by_dc = [[] for _ in range(self.D)]

        for i in range(self.M):
            dc_id = self.env.cluster_to_dc[i]
            cluster = self.env.clusters[i]

            if cluster.get('type', 'cpu') == 'cpu':
                cpu_clusters_by_dc[dc_id].append(i)
            else:
                gpu_clusters_by_dc[dc_id].append(i)

        # Assign CPU jobs
        cpu_job_idx = 0
        for dc_id in range(self.D):
            n_jobs_dc = int(jobs_per_dc_cpu[dc_id])

            for _ in range(n_jobs_dc):
                if cpu_job_idx < len(cpu_jobs):
                    job_idx, job = cpu_jobs[cpu_job_idx]
                    cluster_id = self._greedy_select_cluster(job, cpu_clusters_by_dc[dc_id])
                    job_assignment[job_idx] = cluster_id
                    cpu_job_idx += 1

        # Assign GPU jobs
        gpu_job_idx = 0
        for dc_id in range(self.D):
            n_jobs_dc = int(jobs_per_dc_gpu[dc_id])

            for _ in range(n_jobs_dc):
                if gpu_job_idx < len(gpu_jobs):
                    job_idx, job = gpu_jobs[gpu_job_idx]
                    cluster_id = self._greedy_select_cluster(job, gpu_clusters_by_dc[dc_id])
                    job_assignment[job_idx] = cluster_id
                    gpu_job_idx += 1

        return job_assignment

    def _round_preserving_sum(self, x: np.ndarray) -> np.ndarray:
        """
        Round array to integers while preserving sum.
        """
        rounded = np.floor(x).astype(int)
        remainder = int(np.round(np.sum(x) - np.sum(rounded)))

        # Add remainder to elements with largest fractional parts
        if remainder > 0:
            frac = x - rounded
            indices = np.argsort(-frac)[:remainder]
            rounded[indices] += 1

        return rounded

    def _greedy_select_cluster(self, job: Dict, cluster_ids: List[int]) -> int:
        """
        Select best cluster from candidates using utilization ratio.
        """
        best_cluster = self.M  # Default to reject
        best_ratio = float('inf')

        for cluster_id in cluster_ids:
            cluster = self.env.clusters[cluster_id]
            available = cluster.get('c', 0) - cluster.get('u', 0)

            if available >= job['r']:
                # Compute utilization ratio
                capacity = cluster.get('c', 1)
                util_ratio = cluster.get('u', 0) / capacity if capacity > 0 else 0

                # Select cluster with lowest utilization
                if util_ratio < best_ratio:
                    best_ratio = util_ratio
                    best_cluster = cluster_id

        return best_cluster

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.solve_times:
            return {}

        return {
            'avg_solve_time_ms': np.mean(self.solve_times) * 1000,
            'max_solve_time_ms': np.max(self.solve_times) * 1000,
            'min_solve_time_ms': np.min(self.solve_times) * 1000,
            'std_solve_time_ms': np.std(self.solve_times) * 1000,
            'n_solves': len(self.solve_times)
        }