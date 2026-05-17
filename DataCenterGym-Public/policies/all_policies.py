"""
Scheduling policies for DataCenterGym.

This module implements various job scheduling policies:
- Random Baseline: Random assignment (baseline)
- Greedy Capacity: Assign to cluster with most available capacity
"""
import numpy as np


class SchedulingPolicy:
    """Base class for scheduling policies."""

    def __init__(self, env):
        """
        Args:
            env: DataCenterEnv instance
        """
        self.env = env

    def select_action(self, obs):
        """
        Select action based on current observation.

        Args:
            obs: Current environment observation

        Returns:
            action: Dictionary with job_assignment and T_target
        """
        raise NotImplementedError


class RandomBaseline(SchedulingPolicy):
    """
    Random Baseline Policy.

    Randomly assigns jobs to valid clusters (respecting hardware affinity).
    This serves as a baseline to compare against more sophisticated policies.

    Characteristics:
    - No state awareness
    - Pure random assignment
    - Respects hardware affinity constraints
    - Fixed temperature setpoints
    """

    def __init__(self, env, target_temp=22.0):
        """
        Args:
            env: DataCenterEnv instance
            target_temp: Fixed target temperature for all datacenters
        """
        super().__init__(env)
        self.target_temp = target_temp

    def select_action(self, obs):
        """Random job assignment."""
        job_assignment = []

        for job in self.env.current_jobs:
            # Get valid clusters for this job type
            if self.env.config.get('enforce_affinity', False):
                job_type = job.get('tau', 'cpu')
                valid_clusters = [
                    i for i in range(self.env.M)
                    if self.env.config['cluster_params'][i].get('type', 'cpu') == job_type
                ]
            else:
                valid_clusters = list(range(self.env.M))

            # Randomly assign to valid cluster (always accept)
            if valid_clusters:
                cluster_id = np.random.choice(valid_clusters)
            else:
                cluster_id = self.env.M  # Reject only if no valid clusters

            job_assignment.append(cluster_id)

        # Fixed target temperatures
        T_target = np.array([self.target_temp] * self.env.D, dtype=np.float32)

        return {"job_assignment": job_assignment, "T_target": T_target}


class GreedyCapacity(SchedulingPolicy):
    """
    Greedy Capacity-Based Policy (Load-Balanced).

    Assigns each job to the cluster with the LOWEST utilization ratio
    (among valid clusters respecting hardware affinity). This ensures
    load is distributed evenly across clusters.

    Characteristics:
    - Greedy: Makes locally optimal decisions
    - Load-balanced: Picks least loaded cluster (by utilization %)
    - Avoids cluster starvation by distributing load evenly
    - Does not consider thermal or power constraints explicitly

    This represents a common heuristic used in production systems.
    """

    def __init__(self, env, target_temp=22.0):
        """
        Args:
            env: DataCenterEnv instance
            target_temp: Fixed target temperature for all datacenters
        """
        super().__init__(env)
        self.target_temp = target_temp

    def select_action(self, obs):
        """Greedy assignment to cluster with lowest utilization ratio."""
        job_assignment = []

        for job in self.env.current_jobs:
            # Get valid clusters for this job type
            if self.env.config.get('enforce_affinity', False):
                job_type = job.get('tau', 'cpu')
                valid_clusters = [
                    i for i in range(self.env.M)
                    if self.env.config['cluster_params'][i].get('type', 'cpu') == job_type
                ]
            else:
                valid_clusters = list(range(self.env.M))

            if not valid_clusters:
                job_assignment.append(self.env.M)  # Reject
                continue

            # Find cluster with lowest utilization ratio (most headroom)
            # This ensures load balancing instead of concentrating on large clusters

            # Collect all candidates that can fit the job
            candidates = []
            for cluster_id in valid_clusters:
                cluster = self.env.clusters[cluster_id]
                available = cluster['c'] - cluster['u']

                if available >= job['r']:
                    # Calculate utilization ratio (0.0 = empty, 1.0 = full)
                    if cluster['c'] > 0:
                        util_ratio = cluster['u'] / cluster['c']
                    else:
                        util_ratio = 0.0

                    queue_len = cluster['q']
                    candidates.append((cluster_id, util_ratio, queue_len))

            if candidates:
                # Find minimum utilization ratio
                min_util = min(c[1] for c in candidates)

                # Filter to clusters with minimum utilization
                best_candidates = [c for c in candidates if c[1] == min_util]

                if len(best_candidates) > 1:
                    # Multiple clusters tied - use queue length
                    min_queue = min(c[2] for c in best_candidates)
                    best_candidates = [c for c in best_candidates if c[2] == min_queue]

                if len(best_candidates) > 1:
                    # Still tied - pick deterministically by smallest cluster ID for reproducibility
                    best_cluster = min([c[0] for c in best_candidates])
                else:
                    best_cluster = best_candidates[0][0]

                job_assignment.append(best_cluster)
            else:
                # No cluster has enough capacity, assign anyway to queue
                # Pick the least loaded cluster even if job doesn't fit immediately
                best_cluster = min(valid_clusters,
                                   key=lambda cid: (self.env.clusters[cid]['u'] / self.env.clusters[cid]['c']
                                   if self.env.clusters[cid]['c'] > 0 else 0.0, self.env.clusters[cid]['q']))
                job_assignment.append(best_cluster)

        # Fixed target temperatures
        T_target = np.array([self.target_temp] * self.env.D, dtype=np.float32)

        return {"job_assignment": job_assignment, "T_target": T_target}


class GreedyCapacityThermalAware(SchedulingPolicy):
    """
    Thermal-Aware Greedy Capacity Policy.

    Similar to GreedyCapacity, but also considers datacenter temperature.
    Prefers cooler datacenters when multiple clusters have similar capacity.

    Characteristics:
    - Capacity-aware: Considers available capacity
    - Thermal-aware: Prefers cooler datacenters
    - Weighted decision: Balances capacity and temperature
    """

    def __init__(self, env, target_temp=22.0, thermal_weight=0.3, thermal_target=None):
        """
        Args:
            env: DataCenterEnv instance
            target_temp: Fixed target temperature for all datacenters (cooling setpoint)
            thermal_weight: Weight for thermal consideration (0=capacity only, 1=thermal only)
            thermal_target: Temperature threshold to avoid (defaults to theta_soft if None)
        """
        super().__init__(env)
        self.target_temp = target_temp
        self.thermal_weight = thermal_weight
        self.thermal_target = thermal_target  # If None, use theta_soft

    def select_action(self, obs):
        """Greedy assignment considering both capacity and temperature."""
        job_assignment = []

        for job in self.env.current_jobs:
            # Get valid clusters for this job type
            if self.env.config.get('enforce_affinity', False):
                job_type = job.get('tau', 'cpu')
                valid_clusters = [
                    i for i in range(self.env.M)
                    if self.env.config['cluster_params'][i].get('type', 'cpu') == job_type
                ]
            else:
                valid_clusters = list(range(self.env.M))

            if not valid_clusters:
                job_assignment.append(self.env.M)  # Reject
                continue

            # Score each cluster
            best_cluster = None
            best_score = -float('inf')
            fallback_cluster = None
            fallback_score = -float('inf')

            for cluster_id in valid_clusters:
                cluster = self.env.clusters[cluster_id]
                dc_id = self.env.cluster_to_dc[cluster_id]
                dc = self.env.datacenters[dc_id]

                # Available capacity
                available = cluster['c'] - cluster['u']

                # Check if job fits
                fits = available >= job['r']

                # Capacity score (normalized)
                capacity_score = available / cluster['c'] if cluster['c'] > 0 else 0.0

                # Temperature score (lower is better, normalized)
                # Use thermal_target if specified, otherwise use theta_soft
                if self.thermal_target is not None:
                    thermal_threshold = self.thermal_target
                else:
                    thermal_threshold = self.env.config['datacenter_params'][dc_id].get('thermal_throttling', {}).get('theta_soft', 27.0)

                # Score based on margin below thermal threshold
                temp_margin = (thermal_threshold - dc['theta']) / thermal_threshold
                thermal_score = temp_margin

                # Combined score
                score = (1 - self.thermal_weight) * capacity_score + self.thermal_weight * thermal_score

                if fits:
                    # Job fits - prefer this cluster
                    if score > best_score:
                        best_score = score
                        best_cluster = cluster_id
                else:
                    # Job doesn't fit - track as fallback
                    if score > fallback_score:
                        fallback_score = score
                        fallback_cluster = cluster_id

            # Assign to best fitting cluster, or fallback if none fit
            if best_cluster is not None:
                job_assignment.append(best_cluster)
            elif fallback_cluster is not None:
                job_assignment.append(fallback_cluster)
            else:
                job_assignment.append(self.env.M)  # Reject (only if no valid clusters)

        # Fixed target temperatures
        T_target = np.array([self.target_temp] * self.env.D, dtype=np.float32)

        return {"job_assignment": job_assignment, "T_target": T_target}


class PowerCoolingAware(SchedulingPolicy):
    """
    Power and Cooling-Aware Policy.

    Assigns jobs to minimize total marginal power cost including both
    compute power and cooling power requirements.

    Based on the formula:
        Δp_i(j) = φ_i * r_j + ψ_i * Φ^cool_i(j)

    Where:
    - φ_i * r_j is the compute power cost
    - ψ_i * Φ^cool_i(j) is the estimated incremental cooling cost

    Characteristics:
    - Power-aware: Considers compute power consumption
    - Cooling-aware: Estimates cooling requirements
    - Energy-efficient: Minimizes total energy cost
    - Accounts for temperature impact on cooling
    """

    def __init__(self, env, target_temp=22.0, cooling_weight=1.0):
        """
        Args:
            env: DataCenterEnv instance
            target_temp: Fixed target temperature for all datacenters
            cooling_weight: Weight for cooling cost relative to compute power (ψ_i)
        """
        super().__init__(env)
        self.target_temp = target_temp
        self.cooling_weight = cooling_weight

    def _estimate_cooling_power(self, cluster_id, job, dc):
        """
        Estimate incremental cooling power required if job is assigned.

        Uses a simple model based on:
        - Current temperature deviation from target
        - Expected heat generation from the job
        - Datacenter thermal parameters

        Args:
            cluster_id: Cluster index
            job: Job dictionary
            dc: Datacenter state dictionary

        Returns:
            Estimated incremental cooling power (W)
        """
        # Get cluster and datacenter parameters
        cluster_params = self.env.config['cluster_params'][cluster_id]
        dc_id = self.env.cluster_to_dc[cluster_id]
        dc_params = self.env.config['datacenter_params'][dc_id]

        # Heat generated by this job
        heat_from_job = cluster_params['alpha'] * job['r']

        # Current temperature and target
        current_temp = dc['theta']
        target = dc['T_target']

        # Estimate temperature rise if job is assigned
        # Using RC thermal model: ΔT ≈ R * Q
        R = dc_params['R']
        predicted_temp_rise = R * heat_from_job

        # Cooling needed to maintain target temperature
        # Proportional to temperature deviation + expected rise
        temp_deviation = max(0, current_temp - target + predicted_temp_rise)

        # Simple proportional cooling estimate
        # More deviation = more cooling needed
        cooling_gain = 1500.0  # W per degree (similar to proportional controller)
        estimated_cooling = cooling_gain * temp_deviation

        return estimated_cooling

    def select_action(self, obs):
        """Greedy assignment minimizing total power cost (compute + cooling)."""
        job_assignment = []

        for job in self.env.current_jobs:
            # Get valid clusters for this job type
            if self.env.config.get('enforce_affinity', False):
                job_type = job.get('tau', 'cpu')
                valid_clusters = [
                    i for i in range(self.env.M)
                    if self.env.config['cluster_params'][i].get('type', 'cpu') == job_type
                ]
            else:
                valid_clusters = list(range(self.env.M))

            if not valid_clusters:
                job_assignment.append(self.env.M)  # Reject
                continue

            # Score each cluster by total power cost
            best_cluster = None
            min_power_cost = float('inf')

            # Separate candidates into fitting and non-fitting
            fitting_candidates = []
            fallback_candidates = []

            for cluster_id in valid_clusters:
                cluster = self.env.clusters[cluster_id]
                cluster_params = self.env.config['cluster_params'][cluster_id]
                dc_id = self.env.cluster_to_dc[cluster_id]
                dc = self.env.datacenters[dc_id]

                # Available capacity
                available = cluster['c'] - cluster['u']

                # Compute power cost: α_i * r_j
                compute_power = cluster_params['alpha'] * job['r']

                # Cooling power cost: ψ_i * Φ^cool_i(j)
                cooling_power = self._estimate_cooling_power(cluster_id, job, dc)
                weighted_cooling_power = self.cooling_weight * cooling_power

                # Total marginal power cost
                total_power_cost = compute_power + weighted_cooling_power

                if available >= job['r']:
                    fitting_candidates.append((cluster_id, total_power_cost))
                else:
                    fallback_candidates.append((cluster_id, total_power_cost))

            # Prefer fitting clusters, fall back to any cluster if none fit
            if fitting_candidates:
                best_cluster = min(fitting_candidates, key=lambda x: x[1])[0]
            elif fallback_candidates:
                # No cluster fits - assign to lowest power cost cluster anyway
                best_cluster = min(fallback_candidates, key=lambda x: x[1])[0]
            else:
                best_cluster = None

            if best_cluster is not None:
                job_assignment.append(best_cluster)
            else:
                job_assignment.append(self.env.M)  # Reject (only if no valid clusters)

        # Fixed target temperatures
        T_target = np.array([self.target_temp] * self.env.D, dtype=np.float32)

        return {"job_assignment": job_assignment, "T_target": T_target}


def create_policy(policy_name, env, **kwargs):
    """
    Factory function to create scheduling policy.

    Args:
        policy_name: Name of policy ("random", "greedy_capacity", "greedy_thermal", "power_cooling", "scmpc_nominal", "scmpc_scheduler")
        env: DataCenterEnv instance
        **kwargs: Policy-specific parameters

    Returns:
        SchedulingPolicy instance
    """
    if policy_name.lower() == "random":
        return RandomBaseline(env, **kwargs)
    elif policy_name.lower() == "greedy_capacity":
        return GreedyCapacity(env, **kwargs)
    elif policy_name.lower() == "greedy_thermal":
        return GreedyCapacityThermalAware(env, **kwargs)
    elif policy_name.lower() == "power_cooling":
        return PowerCoolingAware(env, **kwargs)
    elif policy_name.lower() == "scmpc_nominal":
        from policies.all_policies_mpc import SCMPCNominal
        return SCMPCNominal(env, **kwargs)
    elif policy_name.lower() == "scmpc_scheduler":
        from policies.all_policies_mpc_scheduler import SCMPCScheduler
        return SCMPCScheduler(env, **kwargs)
    elif policy_name.lower() == "dc_ratio_mpc":
        from policies.all_policies_dc_ratio_mpc import DCRatioMPCScheduler
        return DCRatioMPCScheduler(env, **kwargs)
    elif policy_name.lower() == "hierarchical_mpc":
        from policies.all_policies_hierarchical_mpc import HierarchicalMPCScheduler
        return HierarchicalMPCScheduler(env, **kwargs)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
