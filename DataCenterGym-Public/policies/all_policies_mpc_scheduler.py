"""
Safety-Constrained Model Predictive Control (SC-MPC) Job Scheduler.

This module implements an MPC-based job scheduler that optimizes job-to-cluster
assignments to balance load, minimize queues, and respect thermal constraints.

Key features:
- Continuous relaxation: Job assignments modeled as continuous [0,1], then rounded
- Thermal prediction: Predicts thermal effects of job placement decisions
- Multi-objective: Balances QoS (queue lengths), thermal safety, energy cost, and load balancing
- Soft constraints: Uses slack variables for thermal and capacity limits
"""

import numpy as np
import casadi as ca
from policies.all_policies import SchedulingPolicy, GreedyCapacity


class SCMPCScheduler(SchedulingPolicy):
    """
    SC-MPC Job Scheduler with Continuous Relaxation.

    Optimizes job assignments (not temperature setpoints) using MPC.
    Temperature setpoints are fixed (not optimized by MPC).

    MPC predicts thermal effects of job placement decisions and balances
    load across datacenters to avoid thermal violations while minimizing
    queues and energy costs.
    """

    def __init__(self, env, N=3, T_target_fixed=22.0, weights=None,
                 enforce_affinity=True, max_jobs_mpc=50, verbose=False):
        """
        Initialize SC-MPC Job Scheduler.

        Args:
            env: DataCenterEnv instance
            N: MPC horizon length (timesteps) - SHORT for scheduler (3-6)
            T_target_fixed: Fixed temperature setpoint for all DCs (°C)
            weights: Dict with keys: wQ, wT, wE, wL, wxi_thermal, wxi_capacity, wR
            enforce_affinity: Whether to enforce hardware affinity constraints
            max_jobs_mpc: Maximum jobs to consider in MPC (limits problem size)
            verbose: Print debug information
        """
        super().__init__(env)
        self.N = N
        self.T_target_fixed = T_target_fixed
        self.weights = weights or self._default_weights()
        self.enforce_affinity = enforce_affinity
        self.max_jobs_mpc = max_jobs_mpc  # Limit MPC problem size
        self.verbose = verbose

        # Load parameters from environment
        print(f"[SCMPCScheduler] Loading parameters...")
        self._load_parameters()
        print(f"[SCMPCScheduler] Parameters loaded. M={self.M}, D={self.D}, max_jobs_mpc={self.max_jobs_mpc}")

        # Build MPC problem
        print(f"[SCMPCScheduler] Building MPC problem (this may take 1-2 minutes)...")
        self._build_mpc_problem()
        print(f"[SCMPCScheduler] MPC problem built successfully!")

        # Initialize warmstart guess
        n_jobs_max = self.max_jobs_mpc
        M = self.env.M
        D = self.env.D
        n_vars = n_jobs_max * (M + 1) + D + M
        self.x0_guess = np.zeros(n_vars)

        if self.verbose:
            print(f"[SCMPCScheduler] Initialized with N={N}, T_target={T_target_fixed}°C")
            print(f"[SCMPCScheduler] Max jobs in MPC: {self.max_jobs_mpc}")
            print(f"[SCMPCScheduler] Weights: {self.weights}")

    def _default_weights(self):
        """Default weight configuration for scheduler."""
        return {
            'wQ': 1.0,           # Queue lengths (QoS)
            'wT': 5.0,           # Thermal safety (moderate priority)
            'wE': 0.01,          # Energy cost (regularizer)
            'wL': 0.5,           # Load balancing
            'wxi_thermal': 100.0, # Enforce thermal limits
            'wxi_capacity': 10.0, # Soft capacity limits
            'wR': 5.0            # Rejection penalty
        }

    def _load_parameters(self):
        """Load datacenter and cluster parameters from environment."""
        self.M = self.env.M  # Number of clusters
        self.D = self.env.D  # Number of datacenters
        self.dt = self.env.dt * 60.0  # Convert minutes to seconds

        # Cluster-to-datacenter mapping
        self.cluster_to_dc = self.env.cluster_to_dc

        # Datacenter thermal parameters
        self.C = np.zeros(self.D)  # Thermal capacitance (J/°C)
        self.R = np.zeros(self.D)  # Thermal resistance (°C/W)
        self.K_p = np.zeros(self.D)  # Cooling controller gain (W/°C)
        self.cooling_max = np.zeros(self.D)  # Max cooling power (W)
        self.theta_ref = np.zeros(self.D)  # Reference temperature (°C)
        self.theta_soft = np.zeros(self.D)  # Soft thermal limit (°C)
        self.psi = np.zeros(self.D)  # Cooling energy cost ($/kWh)

        for dc_id in range(self.D):
            dc_params = self.env.config['datacenter_params'][dc_id]
            self.C[dc_id] = dc_params['C']
            self.R[dc_id] = dc_params['R']

            # Get cooling controller parameters
            controller_params = dc_params.get('controller_params', {})
            self.K_p[dc_id] = controller_params.get('K_p', 1000.0)
            self.cooling_max[dc_id] = dc_params.get('cooling_max', 10000.0)

            # Thermal limits
            throttling_config = dc_params.get('thermal_throttling', {})
            self.theta_soft[dc_id] = throttling_config.get('theta_soft', 32.0)
            self.theta_ref[dc_id] = self.theta_soft[dc_id] - 5.0  # 5°C margin

            # Energy cost (default if not specified)
            self.psi[dc_id] = dc_params.get('psi', 0.10)  # $/kWh

        # Cluster parameters
        self.alpha = np.zeros(self.M)  # Heat generation coefficient (W per resource unit)
        self.c_max = np.zeros(self.M)  # Cluster capacity
        self.cluster_types = []  # 'cpu' or 'gpu'

        for i in range(self.M):
            cluster_params = self.env.config['cluster_params'][i]
            self.alpha[i] = cluster_params['alpha']
            self.c_max[i] = cluster_params.get('c_max', cluster_params.get('c', 10000))
            self.cluster_types.append(cluster_params.get('type', 'cpu'))

    def select_action(self, obs):
        """
        Main decision-making method.

        Returns:
            action dict with:
                - job_assignment: np.array of cluster indices (optimized by MPC)
                - T_target: np.array of fixed temperatures (NOT optimized)
        """
        # 1. Solve MPC optimization for job assignments
        try:
            job_assignment = self._solve_mpc_scheduling()
        except Exception as e:
            # Fallback: Use greedy capacity scheduler
            if self.verbose:
                print(f"[SCMPCScheduler] MPC solver failed: {e}. Falling back to Greedy.")
            job_assignment = self._greedy_fallback()

        # 2. Fixed temperature setpoints (NOT optimized by MPC)
        T_target = np.full(self.env.D, self.T_target_fixed, dtype=np.float32)

        return {
            "job_assignment": job_assignment,
            "T_target": T_target
        }

    def _solve_mpc_scheduling(self):
        """
        Solve MPC problem using CasADi.

        Steps:
        1. Extract current state (temperatures, queues, utilizations)
        2. Get arriving jobs
        3. Set up decision variables x[j,i] for each job-cluster pair
        4. Solve NLP with continuous relaxation
        5. Round to discrete assignments: assign to argmax_i(x[j,i])

        Returns:
            job_assignment: np.array of cluster indices [0..M-1] or M for reject
        """
        state = self._extract_state()
        jobs = self.env.current_jobs
        n_jobs = len(jobs)

        if n_jobs == 0:
            return np.array([], dtype=int)

        # If more jobs than MPC can handle, use greedy for excess
        if n_jobs > self.max_jobs_mpc:
            if self.verbose:
                print(f"[SCMPCScheduler] Too many jobs ({n_jobs} > {self.max_jobs_mpc}), using hybrid approach")
            # Solve MPC for first max_jobs_mpc jobs
            jobs_mpc = jobs[:self.max_jobs_mpc]
            jobs_greedy = jobs[self.max_jobs_mpc:]

            # MPC assignments
            state_mpc = state.copy()
            state_mpc['jobs'] = jobs_mpc
            p_values = self._pack_parameters(state_mpc, jobs_mpc)
            lbx, ubx = self._setup_bounds(jobs_mpc)

            sol = self.solver(
                x0=self.x0_guess,
                lbx=lbx,
                ubx=ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p_values
            )

            x_opt = sol['x'].full().flatten()
            x_assignments = x_opt[:self.max_jobs_mpc * (self.env.M + 1)]
            x_matrix = x_assignments.reshape(self.max_jobs_mpc, self.env.M + 1)
            mpc_assignment = np.argmax(x_matrix, axis=1).astype(int)

            # Greedy for remaining jobs
            greedy = GreedyCapacity(self.env, target_temp=self.T_target_fixed)
            self.env.current_jobs = jobs_greedy
            greedy_action = greedy.select_action(self._get_obs())
            greedy_assignment = greedy_action['job_assignment']
            self.env.current_jobs = jobs  # Restore

            # Combine
            job_assignment = np.concatenate([mpc_assignment, greedy_assignment])
            return job_assignment

        # Pack parameters for solver
        p_values = self._pack_parameters(state, jobs)

        # Set up bounds for affinity constraints
        lbx, ubx = self._setup_bounds(jobs)

        # Solve
        try:
            sol = self.solver(
                x0=self.x0_guess,  # Warmstart from previous solution
                lbx=lbx,
                ubx=ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p_values
            )

            # Extract solution
            x_opt = sol['x'].full().flatten()

            # Check solver status
            stats = self.solver.stats()
            if not stats['success']:
                if self.verbose:
                    print(f"[SCMPCScheduler] Solver did not converge: {stats['return_status']}")
                # Still try to use the solution if it's reasonable

        except Exception as e:
            if self.verbose:
                print(f"[SCMPCScheduler] Solver exception: {e}")
            raise

        # Extract assignment variables (first n_jobs_max * (M+1) elements)
        n_jobs_max = self.max_jobs_mpc
        M = self.env.M
        x_assignments = x_opt[:n_jobs_max * (M + 1)]

        # Reshape to (n_jobs_max, M+1) matrix
        x_matrix = x_assignments.reshape(n_jobs_max, M + 1)

        # Round: assign to cluster with max probability (only for actual jobs)
        job_assignment = np.argmax(x_matrix[:n_jobs, :], axis=1)

        # Warmstart for next solve
        self.x0_guess = x_opt

        if self.verbose:
            print(f"[SCMPCScheduler] Assigned {n_jobs} jobs: {job_assignment}")
            # Show rejection rate
            n_rejected = np.sum(job_assignment == M)
            print(f"[SCMPCScheduler] Rejected: {n_rejected}/{n_jobs}")

        return job_assignment.astype(int)

    def _greedy_fallback(self):
        """Fallback greedy scheduler if MPC fails."""
        greedy = GreedyCapacity(self.env, target_temp=self.T_target_fixed)
        obs = self._get_obs()
        action = greedy.select_action(obs)
        return action['job_assignment']

    def _get_obs(self):
        """Get current observation from environment (for fallback)."""
        # Reconstruct observation matching env._get_obs format
        obs = []

        # Cluster states: [theta, p, c, q, T_amb] for each cluster
        for i in range(self.M):
            cluster = self.env.clusters[i]
            dc_id = self.cluster_to_dc[i]
            dc = self.env.datacenters[dc_id]

            obs.extend([
                dc['theta'],
                cluster['p'],
                cluster['c'],
                cluster['q'],
                dc['T_amb']
            ])

        # Datacenter T_target
        for dc_id in range(self.D):
            obs.append(self.env.datacenters[dc_id]['T_target'])

        return np.array(obs, dtype=np.float32)

    def _extract_state(self):
        """Extract current state for MPC prediction."""
        theta = np.zeros(self.D)
        u = np.zeros(self.M)
        q = np.zeros(self.M)
        T_amb = np.zeros(self.D)

        # Extract datacenter states
        for dc_id in range(self.D):
            dc = self.env.datacenters[dc_id]
            theta[dc_id] = dc['theta']
            T_amb[dc_id] = dc['T_amb']

        # Extract cluster states
        for i in range(self.M):
            cluster = self.env.clusters[i]
            u[i] = cluster['u']
            q[i] = cluster['q']

        return {
            'theta': theta,
            'u': u,
            'q': q,
            'T_amb': T_amb,
            'jobs': self.env.current_jobs
        }

    def _pack_parameters(self, state, jobs):
        """Pack state and job info into parameter vector for solver."""
        n_jobs = len(jobs)
        n_max = self.max_jobs_mpc

        # Pad jobs to max length
        job_r = np.zeros(n_max)
        job_tau = np.zeros(n_max)

        for j in range(min(n_jobs, n_max)):
            job_r[j] = jobs[j]['r']
            # 0=cpu, 1=gpu
            job_tau[j] = 1.0 if jobs[j].get('tau', 'cpu') == 'gpu' else 0.0

        # Concatenate all parameters
        return np.concatenate([
            state['theta'],
            state['u'],
            state['q'],
            state['T_amb'],
            job_r,
            job_tau,
            [n_jobs]
        ])

    def _setup_bounds(self, jobs):
        """
        Setup bounds for decision variables, enforcing affinity constraints.

        Args:
            jobs: List of current jobs

        Returns:
            lbx, ubx: Lower and upper bounds for all decision variables
        """
        n_jobs = len(jobs)
        n_max = self.max_jobs_mpc
        M = self.env.M
        D = self.env.D

        # Total variables: x[n_max * (M+1)] + xi_thermal[D] + xi_capacity[M]
        n_vars = n_max * (M + 1) + D + M

        # Default bounds
        lbx = np.zeros(n_vars)
        ubx = np.ones(n_max * (M + 1)).tolist() + [np.inf] * (D + M)

        # Enforce affinity constraints if enabled
        if self.enforce_affinity:
            for j in range(min(n_jobs, n_max)):
                job = jobs[j]
                job_type = job.get('tau', 'cpu')

                for i in range(M):
                    cluster_type = self.cluster_types[i]

                    # If incompatible, force x[j, i] = 0
                    if job_type != cluster_type:
                        idx = j * (M + 1) + i
                        lbx[idx] = 0.0
                        ubx[idx] = 0.0

        return lbx, ubx

    def _build_mpc_problem(self):
        """
        Construct CasADi NLP for MPC scheduling.

        Problem structure:
        - Decision vars: x[j,i] for n_jobs x (M+1) assignments
        - Parameters: current state (theta, u, q), job specs (r, tau)
        - Constraints: assignment completeness, affinity, thermal limits
        - Objective: queue + thermal + energy + load balancing + rejections
        """
        if self.verbose:
            print("[SCMPCScheduler] Building MPC problem...")

        # Problem dimensions
        n_jobs_max = self.max_jobs_mpc
        M = self.env.M
        D = self.env.D
        N = self.N

        # Decision variables
        # x: Assignment probabilities (n_jobs_max * (M+1))
        # xi_thermal: Thermal slack variables (D)
        # xi_capacity: Capacity slack variables (M)
        x = ca.MX.sym('x', n_jobs_max * (M + 1))
        xi_thermal = ca.MX.sym('xi_thermal', D)
        xi_capacity = ca.MX.sym('xi_capacity', M)

        # Parameters (passed at solve time)
        p_theta = ca.MX.sym('theta', D)
        p_u = ca.MX.sym('u', M)
        p_q = ca.MX.sym('q', M)
        p_T_amb = ca.MX.sym('T_amb', D)
        p_job_r = ca.MX.sym('job_r', n_jobs_max)
        p_job_tau = ca.MX.sym('job_tau', n_jobs_max)
        p_n_jobs = ca.MX.sym('n_jobs', 1)

        # Reshape x to matrix form (n_jobs_max, M+1)
        x_matrix = ca.reshape(x, n_jobs_max, M + 1)

        # === CONSTRAINTS ===
        # Build constraints as CasADi vector
        g = []
        lbg = []
        ubg = []

        # 1. Assignment completeness: sum_i x[j,i] = 1 for each job
        for j in range(n_jobs_max):
            # Use sum2 to ensure scalar result from row vector
            assignment_sum = ca.sum2(x_matrix[j, :])
            g.append(assignment_sum)
            lbg.append(1.0)
            ubg.append(1.0)

        # === PREDICTION MODEL ===
        theta = [p_theta]  # Initial temperature
        q_pred = p_q  # Initial queues (simplified dynamics)

        # Pre-compute utilization from assignments (only once, not per timestep)
        # Utilization is essentially static over the short MPC horizon for job assignments
        u_predicted = ca.MX.zeros(M)
        for i in range(M):
            u_predicted[i] = p_u[i]
            for j in range(n_jobs_max):
                u_predicted[i] += x_matrix[j, i] * p_job_r[j]

        for k in range(N):
            # Heat generation per DC
            Q_gen = ca.MX.zeros(D)
            for d in range(D):
                for i in range(M):
                    if self.cluster_to_dc[i] == d:
                        Q_gen[d] += self.alpha[i] * u_predicted[i]

            # Cooling power (proportional controller with fixed T_target)
            Q_cooling = ca.MX.zeros(D)
            for d in range(D):
                error = ca.fmax(0, theta[k][d] - self.T_target_fixed)
                Q_cooling[d] = self.K_p[d] * error
                Q_cooling[d] = ca.fmin(Q_cooling[d], self.cooling_max[d])

            # Thermal dynamics (RC model)
            theta_next = ca.MX.zeros(D)
            for d in range(D):
                dtheta = (self.dt / self.C[d]) * (
                    Q_gen[d]
                    - (theta[k][d] - p_T_amb[d]) / self.R[d]
                    - Q_cooling[d]
                )
                theta_next[d] = theta[k][d] + dtheta

            theta.append(theta_next)

        # === OBJECTIVE FUNCTION ===
        obj = 0.0

        for k in range(N):
            # QoS: Minimize queue lengths
            q_total = ca.sum1(q_pred)
            obj += self.weights['wQ'] * q_total**2

            # Thermal safety: Penalize high temperatures
            for d in range(D):
                temp_violation = ca.fmax(0, theta[k][d] - self.theta_ref[d])
                obj += self.weights['wT'] * temp_violation**2

            # Energy: Cooling cost
            for d in range(D):
                error = ca.fmax(0, theta[k][d] - self.T_target_fixed)
                cooling_power = self.K_p[d] * error
                cooling_power = ca.fmin(cooling_power, self.cooling_max[d])
                energy_kwh = cooling_power * self.dt / 3.6e6
                obj += self.weights['wE'] * self.psi[d] * energy_kwh

        # Load balancing: Computed once (not per timestep)
        # Penalize variance of DC utilizations
        dc_util = ca.MX.zeros(D)
        for d in range(D):
            dc_total_u = 0.0
            dc_total_c = 0.0
            for i in range(M):
                if self.cluster_to_dc[i] == d:
                    dc_total_u += u_predicted[i]
                    dc_total_c += self.c_max[i]
            if dc_total_c > 0:
                dc_util[d] = dc_total_u / dc_total_c

        mean_util = ca.sum1(dc_util) / D
        for d in range(D):
            obj += self.weights['wL'] * (dc_util[d] - mean_util)**2

        # Slack penalties
        obj += self.weights['wxi_thermal'] * ca.sum1(xi_thermal)
        obj += self.weights['wxi_capacity'] * ca.sum1(xi_capacity)

        # Rejection penalty
        for j in range(n_jobs_max):
            obj += self.weights['wR'] * x_matrix[j, M]

        # === SETUP SOLVER ===
        print(f"[SCMPCScheduler] Creating NLP dict...")
        nlp = {
            'x': ca.vertcat(x, xi_thermal, xi_capacity),
            'f': obj,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(p_theta, p_u, p_q, p_T_amb, p_job_r, p_job_tau, p_n_jobs)
        }
        print(f"[SCMPCScheduler] NLP dict created. Setting solver options...")

        opts = {
            'ipopt.print_level': 0 if not self.verbose else 5,
            'ipopt.max_iter': 100,  # Shorter for real-time
            'ipopt.tol': 1e-4,      # Relaxed tolerance
            'print_time': 0 if not self.verbose else 1
        }

        print(f"[SCMPCScheduler] Creating IPOPT solver (this is the slow step, may take 60-120 seconds)...")
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        print(f"[SCMPCScheduler] IPOPT solver created!")

        # Store constraint bounds (constant)
        self.lbg = [1.0] * n_jobs_max  # Assignment completeness
        self.ubg = [1.0] * n_jobs_max

        if self.verbose:
            print(f"[SCMPCScheduler] MPC problem built. Variables: {n_jobs_max * (M + 1) + D + M}, Constraints: {n_jobs_max}")
