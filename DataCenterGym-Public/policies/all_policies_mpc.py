"""
Safety-Constrained Model Predictive Control (SC-MPC) policy for datacenter optimization.

This module implements SC-MPC for the nominal operating regime (RQ1), optimizing
cooling setpoints while delegating job assignment to a base heuristic policy.
"""
import numpy as np

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False
    print("Warning: CasADi not installed. SC-MPC policies will not be available.")
    print("Install with: pip install casadi")


class SCMPCNominal:
    """
    Safety-Constrained MPC for Nominal Operating Regime (RQ1).

    Optimizes cooling setpoints T_target[d] for each datacenter to minimize
    a multi-objective cost function (QoS, thermal safety, energy) while
    delegating job assignment to a base heuristic policy.

    Features:
    - MPC horizon: N timesteps (default 6)
    - Decision variables: T_target per datacenter (constant over horizon)
    - Prediction model: Datacenter-level thermal dynamics with coarse workload assumption
    - Constraints: Thermal limits, cooling capacity, soft QoS bounds
    - Objective: Weighted sum of queue lengths, thermal headroom, energy cost, and regularizers
    """

    def __init__(self, env, N=6, target_temp_init=22.0,
                 base_policy="greedy_capacity", weights=None,
                 q_max_dc=500.0, solver_verbose=False):
        """
        Initialize SC-MPC policy.

        Args:
            env: DataCenterEnv instance
            N: MPC horizon length (timesteps)
            target_temp_init: Initial T_target for all DCs (°C)
            base_policy: Policy name string or policy instance for job assignment
                        String options: "greedy_capacity", "greedy_thermal", "power_cooling"
                        Instance: Pass pre-created policy object
            weights: Dict with keys: wQ, wT, wE, wxi, wDelta
                    If None, use nominal defaults
            q_max_dc: QoS constraint threshold (jobs per datacenter)
            solver_verbose: If True, print IPOPT solver output
        """
        if not CASADI_AVAILABLE:
            raise ImportError("CasADi is required for SC-MPC policies. Install with: pip install casadi")

        self.env = env
        self.N = N
        self.q_max_dc = q_max_dc
        self.solver_verbose = solver_verbose

        # Create or store base policy for job assignment
        if isinstance(base_policy, str):
            from policies.all_policies import create_policy
            self.base_policy = create_policy(base_policy, env, target_temp=target_temp_init)
        else:
            self.base_policy = base_policy

        # Load system parameters from environment config
        self._load_parameters()

        # Set objective weights (nominal operating point)
        if weights is None:
            self.weights = {
                'wQ': 1.0,      # QoS primary objective
                'wT': 10.0,     # Thermal safety high priority
                'wE': 0.01,     # Energy regularizer (secondary)
                'wxi': 100.0,   # Enforce QoS band tightly
                'wDelta': 1.0   # Smooth setpoint changes
            }
        else:
            self.weights = weights

        # Initialize previous T_target for smoothness penalty
        self.T_prev = np.array([target_temp_init] * self.D, dtype=np.float32)

        # Build MPC optimization problem
        self._build_mpc_problem()

        # Warmstart solution (use previous solution as initial guess)
        self.prev_solution = None

    def _load_parameters(self):
        """Load all datacenter parameters from environment config."""
        self.D = self.env.D
        self.M = self.env.M

        # Thermal parameters per datacenter
        self.R = np.array([
            self.env.config['datacenter_params'][d]['R']
            for d in range(self.D)
        ])

        self.C = np.array([
            self.env.config['datacenter_params'][d]['C']
            for d in range(self.D)
        ])

        self.theta_max = np.array([
            self.env.config['datacenter_params'][d]['theta_max']
            for d in range(self.D)
        ])

        self.cooling_max = np.array([
            self.env.config['datacenter_params'][d]['cooling_max']
            for d in range(self.D)
        ])

        self.K_p = np.array([
            self.env.config['datacenter_params'][d]['controller_params']['K_p']
            for d in range(self.D)
        ])

        # Cluster parameters
        self.alpha = np.array([
            self.env.config['cluster_params'][i]['alpha']
            for i in range(self.M)
        ])

        self.phi = np.array([
            self.env.config['cluster_params'][i]['phi']
            for i in range(self.M)
        ])

        # Cluster to datacenter mapping
        self.cluster_to_dc_map = [
            self.env.cluster_to_dc[i] for i in range(self.M)
        ]

        # Time step (convert minutes to seconds)
        self.dt = self.env.dt * 60  # 5 minutes = 300 seconds

        # Reference temperature for thermal headroom cost (below theta_soft)
        self.theta_ref = np.array([27.0] * self.D)

        # Time-of-use pricing (average price per datacenter)
        self.psi = np.zeros(self.D)
        pricing_config = self.env.config.get('pricing', {})

        if pricing_config.get('per_datacenter', False):
            dc_pricing = pricing_config.get('datacenters', {})
            for d in range(self.D):
                if d in dc_pricing:
                    peak = dc_pricing[d].get('peak_price', 0.15)
                    off_peak = dc_pricing[d].get('offpeak_price', 0.10)
                    self.psi[d] = (peak + off_peak) / 2.0
                else:
                    self.psi[d] = 0.12  # Default average
        else:
            # Global pricing
            peak = pricing_config.get('peak_price', 0.15)
            off_peak = pricing_config.get('offpeak_price', 0.10)
            self.psi[:] = (peak + off_peak) / 2.0

    def _extract_state(self):
        """Extract current state from environment for MPC prediction."""
        # Datacenter thermal state
        theta_current = np.array([
            self.env.datacenters[d]['theta']
            for d in range(self.D)
        ])

        T_amb = np.array([
            self.env.datacenters[d]['T_amb']
            for d in range(self.D)
        ])

        # Current utilization per cluster (for heat generation prediction)
        u_clusters = np.array([
            self.env.clusters[i]['u']
            for i in range(self.M)
        ])

        # Queue lengths (datacenter-level aggregate)
        q_dc = np.zeros(self.D)
        for i in range(self.M):
            dc_id = self.env.cluster_to_dc[i]
            q_dc[dc_id] += self.env.clusters[i]['q']

        # Previous T_target (for smoothness penalty)
        T_prev = np.array([
            self.env.datacenters[d]['T_target']
            for d in range(self.D)
        ])

        return {
            'theta': theta_current,
            'T_amb': T_amb,
            'u_clusters': u_clusters,
            'q_dc': q_dc,
            'T_prev': T_prev
        }

    def _build_mpc_problem(self):
        """Build CasADi optimization problem for MPC."""
        # Decision variables
        Tt = ca.MX.sym('Tt', self.D)         # Cooling setpoints (one per DC)
        xi = ca.MX.sym('xi', self.D)         # Slack variables for QoS
        X = ca.vertcat(Tt, xi)               # Combined decision vector

        # Parameters (will be set at each solve)
        theta_0 = ca.MX.sym('theta_0', self.D)      # Initial temperature
        T_amb = ca.MX.sym('T_amb', self.D)          # Ambient temperature
        u_clusters = ca.MX.sym('u_clusters', self.M)  # Cluster utilization
        q_dc = ca.MX.sym('q_dc', self.D)            # Queue lengths per DC
        T_prev = ca.MX.sym('T_prev', self.D)        # Previous setpoints

        params = ca.vertcat(theta_0, T_amb, u_clusters, q_dc, T_prev)

        # Build objective and constraints
        obj = 0
        g = []  # Constraint expressions
        lbg = []  # Lower bounds on constraints
        ubg = []  # Upper bounds on constraints

        # Initialize thermal state trajectory
        theta = [theta_0]

        # Predict over horizon
        for k in range(self.N):
            # ===== Thermal Dynamics Prediction =====
            # Heat generation per datacenter (aggregate clusters)
            Q_gen = ca.MX.zeros(self.D)
            for i in range(self.M):
                dc_id = self.cluster_to_dc_map[i]
                Q_gen[dc_id] += self.alpha[i] * u_clusters[i]

            # Cooling power (proportional model with CasADi-compatible operations)
            Pc = ca.MX.zeros(self.D)
            for d in range(self.D):
                # Proportional cooling: P_c = K_p * max(0, theta - T_target)
                error = ca.fmax(0, theta[k][d] - Tt[d])
                Pc_unbounded = self.K_p[d] * error
                Pc[d] = ca.fmin(Pc_unbounded, self.cooling_max[d])  # Clip to max

            # Thermal dynamics: theta[k+1] = theta[k] + dt/C * (Q_gen - (theta-T_amb)/R - Pc)
            theta_next = ca.MX.zeros(self.D)
            for d in range(self.D):
                heat_dissipation = (theta[k][d] - T_amb[d]) / self.R[d]
                net_heat = Q_gen[d] - heat_dissipation - Pc[d]
                dtheta = (net_heat / self.C[d]) * self.dt
                theta_next[d] = theta[k][d] + dtheta

            theta.append(theta_next)

            # ===== Thermal Constraints =====
            for d in range(self.D):
                g.append(theta_next[d])
                lbg.append(-ca.inf)
                ubg.append(self.theta_max[d])

            # ===== Objective Function =====
            # QoS: Queue penalty (using current state, constant over horizon)
            obj += self.weights['wQ'] * ca.sum1(q_dc)

            # Thermal headroom: Penalize temps above reference
            for d in range(self.D):
                temp_excess = ca.fmax(0, theta_next[d] - self.theta_ref[d])
                obj += self.weights['wT'] * temp_excess**2

            # Energy cost: Compute + Cooling power
            for d in range(self.D):
                # Compute power (from clusters in this DC)
                P_comp_d = 0
                for i in range(self.M):
                    if self.cluster_to_dc_map[i] == d:
                        P_comp_d += self.phi[i] * u_clusters[i]

                # Cooling power
                P_cool_d = Pc[d]

                # Convert to kWh: (W * seconds) / 3.6e6
                energy_kwh = (P_comp_d + P_cool_d) * self.dt / 3.6e6
                obj += self.weights['wE'] * self.psi[d] * energy_kwh

        # ===== Slack Penalty (L1 norm) =====
        obj += self.weights['wxi'] * ca.sum1(xi)

        # ===== Setpoint Smoothness (L2 regularizer) =====
        obj += self.weights['wDelta'] * ca.sumsqr(Tt - T_prev)

        # ===== QoS Soft Constraints =====
        for d in range(self.D):
            g.append(q_dc[d] - self.q_max_dc - xi[d])
            lbg.append(-ca.inf)
            ubg.append(0)  # q_dc[d] <= q_max_dc + xi[d]

        # ===== Decision Variable Bounds =====
        lbx = [18.0] * self.D + [0.0] * self.D  # Tt: [18, 28], xi: [0, inf]
        ubx = [28.0] * self.D + [ca.inf] * self.D

        # ===== Setup NLP =====
        nlp = {
            'x': X,
            'p': params,
            'f': obj,
            'g': ca.vertcat(*g)
        }

        # IPOPT solver options
        opts = {
            'ipopt.print_level': 5 if self.solver_verbose else 0,
            'ipopt.max_iter': 200,
            'ipopt.tol': 1e-6,
            'ipopt.acceptable_tol': 1e-4,
            'print_time': 0 if not self.solver_verbose else 1
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg
        self.n_params = params.size1()

    def _solve_mpc(self):
        """
        Solve MPC optimization problem.

        Returns:
            T_target_opt: Optimal cooling setpoints (numpy array, shape (D,))
        """
        # Extract current state
        state = self._extract_state()

        # Pack parameters
        p_val = np.concatenate([
            state['theta'],
            state['T_amb'],
            state['u_clusters'],
            state['q_dc'],
            state['T_prev']
        ])

        # Initial guess (warmstart with previous solution if available)
        if self.prev_solution is not None:
            x0 = self.prev_solution
        else:
            # Default initial guess: T_target = T_prev, xi = 0
            x0 = np.concatenate([state['T_prev'], np.zeros(self.D)])

        # Solve
        sol = self.solver(
            x0=x0,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p_val
        )

        # Extract solution
        x_opt = sol['x'].full().flatten()
        self.prev_solution = x_opt  # Save for warmstart

        # Extract T_target (first D elements)
        T_target_opt = x_opt[:self.D]

        # Check solver status
        stats = self.solver.stats()
        if not stats['success']:
            print(f"Warning: MPC solver failed with status: {stats['return_status']}")

        return T_target_opt.astype(np.float32)

    def select_action(self, obs):
        """
        Select action by combining base policy job assignment with MPC cooling setpoints.

        Args:
            obs: Current environment observation

        Returns:
            action: Dictionary with job_assignment and T_target
        """
        # 1. Get job assignments from base policy
        base_action = self.base_policy.select_action(obs)
        job_assignment = base_action["job_assignment"]

        # 2. Solve MPC optimization for T_target
        try:
            T_target_opt = self._solve_mpc()

            # Update T_prev for next timestep
            self.T_prev = T_target_opt.copy()

        except Exception as e:
            # Graceful degradation: Fall back to base policy's T_target
            print(f"MPC solver failed: {e}. Using base policy T_target.")
            T_target_opt = base_action["T_target"]
            self.T_prev = T_target_opt.copy()

        # 3. Return combined action
        return {
            "job_assignment": job_assignment,
            "T_target": T_target_opt
        }
