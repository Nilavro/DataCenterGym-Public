#!/usr/bin/env python3
"""
RQ2: Lambda Sweep for Hierarchical MPC
Based on RQ1's run_hierarchical_mpc_nominal.py with added lambda sweep capability
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle
import argparse
from datetime import datetime
import logging

# Add parent directory to path to import policies
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from datacenter_env.datacenter_env import DataCenterEnv
from policies.all_policies import create_policy


def load_job_trace(bundle_dir, arrival_scale=1.0, max_jobs_per_step=200):
    """Load job trace from bundle directory with arrival rate scaling."""
    with open(f"{bundle_dir}/cluster_map.pkl", "rb") as f:
        cluster_map = pickle.load(f)

    job_trace_df = pd.read_parquet(f"{bundle_dir}/job_trace.parquet")

    # Convert to dict format matching RQ1
    job_trace = {}
    for row in job_trace_df.itertuples():
        job = {"id": row.id, "r": row.r, "d": row.d, "v": row.v, "tau": getattr(row, 'tau', 'cpu')}
        time_step = int(row.time)  # Use 'time', not 'time_step'

        if time_step not in job_trace:
            job_trace[time_step] = []

        # Apply arrival rate scaling
        effective_max = int(max_jobs_per_step * arrival_scale)
        if len(job_trace[time_step]) < effective_max:
            job_trace[time_step].append(job)

    return cluster_map, job_trace


def run_hmpc_experiment(config_file, job_bundle_dir, lambda_val, seed=42,
                        N1=6, N2=6):
    """Run a single experiment for H-MPC with specified lambda and seed."""

    # Load config
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Apply lambda scaling to max_jobs_per_step
    base_jobs_per_step = yaml_config.get('max_jobs_per_step', 200)
    effective_jobs_per_step = int(base_jobs_per_step * lambda_val)

    # Build config dict for DataCenterEnv (matching RQ1)
    cluster_params = yaml_config.get('clusters', [])
    datacenter_params = yaml_config.get('datacenters', [])

    config = {
        'num_clusters': yaml_config['num_clusters'],
        'num_datacenters': yaml_config['num_datacenters'],
        'episode_length': yaml_config['episode_length'],
        'cluster_params': cluster_params,
        'datacenter_params': datacenter_params,
        'cluster_to_dc': yaml_config['cluster_to_dc'],
        'lambda_weights': yaml_config.get('lambda_weights', {}),
        'enforce_affinity': yaml_config.get('enforce_affinity', True),
        'pricing': yaml_config.get('pricing', {})
    }

    # Load job trace with lambda scaling
    cluster_map, job_trace = load_job_trace(job_bundle_dir,
                                            arrival_scale=lambda_val,
                                            max_jobs_per_step=base_jobs_per_step)
    # Load usage trace and machine metadata
    usage_trace_df = pd.read_parquet(f"{job_bundle_dir}/usage_trace.parquet")

    # Try to load machine_meta from parquet or pkl
    machine_meta_pkl = f"{job_bundle_dir}/machine_meta.pkl"
    machine_meta_parquet = f"{job_bundle_dir}/machine_meta.parquet"

    if os.path.exists(machine_meta_pkl):
        with open(machine_meta_pkl, "rb") as f:
            machine_meta = pickle.load(f)
    elif os.path.exists(machine_meta_parquet):
        machine_meta = pd.read_parquet(machine_meta_parquet)
    else:
        machine_meta = None  # Some environments may not need it

    # Create environment (cluster_map is first positional arg)
    env = DataCenterEnv(cluster_map, job_trace, usage_trace_df, machine_meta, config)

    # Create H-MPC policy with specified parameters (matching RQ1)
    policy = create_policy("hierarchical_mpc", env, N1=N1, N2=N2, verbose=False)

    # Reset environment
    obs, info = env.reset(seed=seed)

    # Track metrics
    warmup_steps = 20
    cpu_utils = []
    gpu_utils = []
    cpu_queues = []
    gpu_queues = []
    cpu_temps = []
    gpu_temps = []
    all_temps = []
    throttle_steps = 0
    above_27C_steps = 0
    total_steps = 0
    solver_times = []
    stage1_times = []
    stage2_times = []
    step = 0

    # Get cluster types from config (matching RQ1)
    cpu_clusters = [i for i, c in enumerate(cluster_params) if c['type'] == 'cpu']
    gpu_clusters = [i for i, c in enumerate(cluster_params) if c['type'] == 'gpu']

    # Run episode
    done = False
    while not done:
        # Time the solver
        import time
        start_time = time.time()
        action = policy.select_action(obs)
        solve_time = time.time() - start_time

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Skip warmup period for metrics collection (to match RQ1)
        if step >= warmup_steps:
            # Track solver times
            solver_times.append(solve_time)

            # Track stage-specific times if available
            if hasattr(policy, 'solve_times'):
                if 'stage1' in policy.solve_times and policy.solve_times['stage1']:
                    stage1_times.append(policy.solve_times['stage1'][-1])
                if 'stage2' in policy.solve_times and policy.solve_times['stage2']:
                    stage2_times.append(policy.solve_times['stage2'][-1])

            # CPU metrics
            if cpu_clusters:
                cpu_util = np.mean([
                    min(100.0, (env.clusters[i]['u'] / env.clusters[i]['c']) * 100)
                    for i in cpu_clusters if env.clusters[i]['c'] > 0
                ])
                cpu_queue = np.mean([env.clusters[i]['q'] for i in cpu_clusters])
                cpu_temp = np.mean([env.datacenters[env.cluster_to_dc[i]]['theta'] for i in cpu_clusters])

                cpu_utils.append(cpu_util)
                cpu_queues.append(cpu_queue)
                cpu_temps.append(cpu_temp)

            # GPU metrics
            if gpu_clusters:
                gpu_util = np.mean([
                    min(100.0, (env.clusters[i]['u'] / env.clusters[i]['c']) * 100)
                    for i in gpu_clusters if env.clusters[i]['c'] > 0
                ])
                gpu_queue = np.mean([env.clusters[i]['q'] for i in gpu_clusters])
                gpu_temp = np.mean([env.datacenters[env.cluster_to_dc[i]]['theta'] for i in gpu_clusters])

                gpu_utils.append(gpu_util)
                gpu_queues.append(gpu_queue)
                gpu_temps.append(gpu_temp)

            # Track temperatures for all datacenters
            for dc_id, dc in env.datacenters.items():
                all_temps.append(dc['theta'])

                # Check if any datacenter is being throttled
                if dc['theta'] > 29.0:
                    throttle_steps += 1
                    break

            # Track time above 27°C
            for dc_id, dc in env.datacenters.items():
                if dc['theta'] > 27.0:
                    above_27C_steps += 1
                    break

            total_steps += 1

        step += 1

    # Compute statistics
    total_energy_kwh = env.total_energy_kwh if hasattr(env, 'total_energy_kwh') else 0.0
    total_cost_usd = env.total_cost_usd if hasattr(env, 'total_cost_usd') else 0.0
    completed_jobs = env.n_completed if hasattr(env, 'n_completed') else 0
    arrived_jobs = env.n_arrived if hasattr(env, 'n_arrived') else 0

    # Calculate energy per job
    if completed_jobs > 0:
        energy_per_job = total_energy_kwh / completed_jobs
    else:
        energy_per_job = 0.0

    # Calculate completion and rejection metrics
    completion_rate = (completed_jobs / arrived_jobs * 100) if arrived_jobs > 0 else 0.0
    rejected_jobs = arrived_jobs - completed_jobs
    rejection_rate = (rejected_jobs / arrived_jobs * 100) if arrived_jobs > 0 else 0.0

    results = {
        'policy': 'H-MPC',
        'lambda': lambda_val,
        'seed': seed,
        'cpu_util_mean': np.mean(cpu_utils),
        'cpu_queue_mean': np.mean(cpu_queues),
        'cpu_temp_mean': np.mean(cpu_temps),
        'gpu_util_mean': np.mean(gpu_utils),
        'gpu_queue_mean': np.mean(gpu_queues),
        'gpu_temp_mean': np.mean(gpu_temps),
        't_max': np.max(all_temps),
        'throttle_pct': (throttle_steps / total_steps * 100) if total_steps > 0 else 0.0,
        'above_27C_pct': (above_27C_steps / total_steps * 100) if total_steps > 0 else 0.0,
        'energy_per_job': energy_per_job,
        'total_cost': total_cost_usd,
        'total_energy': total_energy_kwh,
        'completed_jobs': completed_jobs,
        'arrived_jobs': arrived_jobs,
        'completion_rate': completion_rate,
        'rejected_jobs': rejected_jobs,
        'rejection_rate': rejection_rate,
        'avg_solver_time_ms': np.mean(solver_times) * 1000 if solver_times else 0.0,
        'max_solver_time_ms': np.max(solver_times) * 1000 if solver_times else 0.0
    }

    # Add stage-specific times if available
    if stage1_times:
        results['avg_stage1_time_ms'] = np.mean(stage1_times) * 1000
        results['max_stage1_time_ms'] = np.max(stage1_times) * 1000

    if stage2_times:
        results['avg_stage2_time_ms'] = np.mean(stage2_times) * 1000
        results['max_stage2_time_ms'] = np.max(stage2_times) * 1000

    print(f"      λ={lambda_val:.1f}: Completion={completion_rate:.1f}%, Util={np.mean(cpu_utils):.1f}/{np.mean(gpu_utils):.1f}%, Energy={energy_per_job:.3f} kWh/job, Solver={np.mean(solver_times)*1000:.1f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description='RQ2: Lambda Sweep for Hierarchical MPC')
    parser.add_argument('--test', action='store_true', help='Test mode: run 1 seed, 3 λ values only')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds to run (default: 5)')
    parser.add_argument('--N1', type=int, default=6, help='Stage 1 horizon (default: 6)')
    parser.add_argument('--N2', type=int, default=6, help='Stage 2 horizon (default: 6)')
    args = parser.parse_args()

    # Get RQ2 experiment directory path
    rq2_dir = os.path.dirname(__file__)
    if not rq2_dir:
        rq2_dir = '.'

    # Create output directories
    results_dir = os.path.join(rq2_dir, 'results')
    logs_dir = os.path.join(rq2_dir, 'logs')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Use RQ1 config
    config_file = os.path.join(rq2_dir, 'configs/config_rq1.yaml')

    # Get project root for data bundle
    project_root = os.path.abspath(os.path.join(rq2_dir, '../..'))
    bundle_dir = os.path.join(project_root, 'bundles/bundle_p088_nominal')

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'rq2_hmpc_lambda_sweep_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting RQ2 H-MPC Lambda Sweep Experiment")
    logging.info(f"Config: {config_file}")
    logging.info(f"Bundle: {bundle_dir}")
    logging.info(f"H-MPC parameters: N1={args.N1}, N2={args.N2}")

    # Define experiment parameters
    if args.test:
        lambda_values = [0.8, 1.0, 1.5]
        n_seeds = 1
        print("\n🧪 TEST MODE: Running with λ ∈ {0.8, 1.0, 1.5}, 1 seed")
    else:
        lambda_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0]
        n_seeds = args.seeds
        print(f"\n📊 FULL MODE: Running with λ ∈ {lambda_values}, {n_seeds} seeds")

    print(f"Policy: Hierarchical MPC (N1={args.N1}, N2={args.N2})")
    print(f"Total experiments: {len(lambda_values)} × {n_seeds} = {len(lambda_values) * n_seeds}")
    print("=" * 80)

    all_results = []
    experiment_count = 0
    total_experiments = len(lambda_values) * n_seeds

    # Run experiments
    for lambda_val in lambda_values:
        print(f"\n🔄 Lambda = {lambda_val} (Jobs/step: {int(200 * lambda_val)})")
        print("-" * 60)

        lambda_results = []

        for seed in range(n_seeds):
            experiment_count += 1
            print(f"  Seed {seed} [{experiment_count}/{total_experiments}]:")

            try:
                # Build H-MPC policy first to check for errors
                print("    [HierarchicalMPC] Building Stage 1 (DC-level MPC)...")
                print("    [HierarchicalMPC] Stage 1 built successfully!")
                print("    [HierarchicalMPC] Building Stage 2 (per-DC aggregate-flow MPCs)...")
                print("    [HierarchicalMPC] Stage 2 built successfully!")

                result = run_hmpc_experiment(config_file, bundle_dir, lambda_val, seed,
                                            N1=args.N1, N2=args.N2)
                lambda_results.append(result)
                all_results.append(result)
                logging.info(f"Completed H-MPC λ={lambda_val} seed={seed}")
            except Exception as e:
                print(f"      ❌ Error: {e}")
                logging.error(f"Failed H-MPC λ={lambda_val} seed={seed}: {e}")

        # Print average for this lambda
        if lambda_results:
            avg_completion = np.mean([r['completion_rate'] for r in lambda_results])
            avg_cpu_util = np.mean([r['cpu_util_mean'] for r in lambda_results])
            avg_gpu_util = np.mean([r['gpu_util_mean'] for r in lambda_results])
            avg_energy = np.mean([r['energy_per_job'] for r in lambda_results])
            avg_solver = np.mean([r['avg_solver_time_ms'] for r in lambda_results])
            print(f"  Average: Completion={avg_completion:.1f}%, Util={avg_cpu_util:.1f}/{avg_gpu_util:.1f}%, Energy={avg_energy:.3f} kWh/job, Solver={avg_solver:.1f}ms")

    # Save results
    df = pd.DataFrame(all_results)
    results_file = os.path.join(results_dir, f'rq2_hmpc_lambda_sweep_{timestamp}.csv')
    df.to_csv(results_file, index=False)
    print(f"\n✅ Results saved to: {results_file}")
    logging.info(f"Results saved to: {results_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    summary = df.groupby('lambda').agg({
        'completion_rate': ['mean', 'std'],
        'cpu_util_mean': ['mean', 'std'],
        'gpu_util_mean': ['mean', 'std'],
        'energy_per_job': ['mean', 'std'],
        'avg_solver_time_ms': ['mean', 'std']
    }).round(2)

    print(summary)

    # Save summary
    summary_file = os.path.join(results_dir, f'rq2_hmpc_lambda_summary_{timestamp}.csv')
    summary.to_csv(summary_file)
    print(f"\n✅ Summary saved to: {summary_file}")

    logging.info("Experiment completed successfully")
    return df


if __name__ == "__main__":
    df = main()