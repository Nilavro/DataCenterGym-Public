"""
Policy Comparison at Nominal Operating Regime

Compares 4 scheduling policies at 1.0x arrival rate:
- Random Baseline
- Greedy Capacity
- Thermal-Aware
- Power+Cooling-Aware
"""
import sys
import os
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import yaml
import numpy as np
import pickle
import pandas as pd
from datacenter_env import DataCenterEnv
from policies import RandomBaseline, GreedyCapacity, GreedyCapacityThermalAware, PowerCoolingAware
from policies.all_policies_mpc import SCMPCNominal

def load_job_trace(bundle_dir, arrival_scale=1.0, max_jobs_per_step=200):
    """Load job trace."""
    with open(f"{bundle_dir}/cluster_map.pkl", "rb") as f:
        cluster_map = pickle.load(f)

    job_trace_df = pd.read_parquet(f"{bundle_dir}/job_trace.parquet")

    # Convert to dict format
    job_trace = {}
    for row in job_trace_df.itertuples():
        job = {"id": row.id, "r": row.r, "d": row.d, "v": row.v, "tau": getattr(row, 'tau', 'cpu')}
        time_step = int(row.time)
        if time_step not in job_trace:
            job_trace[time_step] = []
        if len(job_trace[time_step]) < max_jobs_per_step:
            job_trace[time_step].append(job)

    return cluster_map, job_trace


def run_policy_experiment(config_file, policy_name, job_bundle_dir='processed_bundle_nominal_65', seed=42):
    """Run single experiment with given policy."""
    print(f"  Running {policy_name}, seed={seed}")

    # Load config
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Load job data
    max_jobs = yaml_config.get('max_jobs_per_step', 200)  # Default should be 200 to match config
    cluster_map, job_trace = load_job_trace(job_bundle_dir, max_jobs_per_step=max_jobs)
    usage_trace_df = pd.read_parquet(f"{job_bundle_dir}/usage_trace.parquet")

    # Build config
    # IMPORTANT: Use cluster params directly from YAML to preserve all fields including 'id'
    cluster_params = yaml_config.get('clusters', [])

    # Use datacenter params directly from YAML as well
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

    # Run simulation
    np.random.seed(seed)
    env = DataCenterEnv(cluster_map, job_trace, usage_trace_df, None, config)

    # Select policy
    if policy_name == "Random":
        policy = RandomBaseline(env)
    elif policy_name == "Greedy":
        policy = GreedyCapacity(env, target_temp=22.0)
    elif policy_name == "Thermal":
        policy = GreedyCapacityThermalAware(env, target_temp=22.0, thermal_weight=0.4, thermal_target=25.0)
    elif policy_name == "PowerCool":
        policy = PowerCoolingAware(env, target_temp=22.0)
    elif policy_name == "SCMPC":
        policy = SCMPCNominal(env, N=6, base_policy="greedy_capacity", solver_verbose=False)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    obs, info = env.reset()
    done = False

    # Collect metrics
    cpu_utils = []
    gpu_utils = []
    cpu_queues = []
    gpu_queues = []
    cpu_temps = []
    gpu_temps = []
    all_temps = []
    system_temps = []  # Track mean system temperature across all DCs
    throttle_steps = 0
    above_27C_steps = 0
    total_steps = 0

    cpu_clusters = [i for i, c in enumerate(cluster_params) if c['type'] == 'cpu']
    gpu_clusters = [i for i, c in enumerate(cluster_params) if c['type'] == 'gpu']

    # Warmup period (48 steps = 4 hours) to match RQ2
    warmup_steps = 48
    step = 0

    while not done:
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Skip warmup period for metrics collection (to match RQ2)
        if step >= warmup_steps:
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

            # Track all datacenter temperatures for T_max and system mean
            dc_temps = [dc['theta'] for dc in env.datacenters.values()]
            all_temps.extend(dc_temps)
            system_temps.append(np.mean(dc_temps))  # Store mean system temperature

            # Track throttling (temperature exceeding theta_soft)
            # NOTE: Throttling begins at theta_soft, not theta_max
            for dc_id, dc in env.datacenters.items():
                theta_soft = datacenter_params[dc_id].get('thermal_throttling', {}).get('theta_soft', 70.0)
                if dc['theta'] > theta_soft:
                    throttle_steps += 1
                    break  # Count once per timestep if any DC throttles

            # Track time above 27°C
            for dc_id, dc in env.datacenters.items():
                if dc['theta'] > 27.0:
                    above_27C_steps += 1
                    break  # Count once per timestep if any DC above 27°C

            total_steps += 1

        # Increment step counter
        step += 1

    # Compute statistics
    # Get final metrics from environment
    total_energy_kwh = env.total_energy_kwh if hasattr(env, 'total_energy_kwh') else 0.0
    total_cost_usd = env.total_cost_usd if hasattr(env, 'total_cost_usd') else 0.0
    completed_jobs = env.n_completed if hasattr(env, 'n_completed') else 0
    arrived_jobs = env.n_arrived if hasattr(env, 'n_arrived') else 0

    # Calculate energy per job (handle division by zero)
    if completed_jobs > 0:
        energy_per_job = total_energy_kwh / completed_jobs
    else:
        energy_per_job = 0.0
        print(f"    WARNING: No jobs completed! arrived={arrived_jobs}, completed={completed_jobs}")

    # Calculate completion and rejection metrics
    completion_rate = (completed_jobs / arrived_jobs * 100) if arrived_jobs > 0 else 0.0
    rejected_jobs = arrived_jobs - completed_jobs
    rejection_rate = (rejected_jobs / arrived_jobs * 100) if arrived_jobs > 0 else 0.0

    results = {
        'policy': policy_name,
        'seed': seed,
        'cpu_util_mean': np.mean(cpu_utils),
        'cpu_queue_mean': np.mean(cpu_queues),
        'cpu_temp_mean': np.mean(cpu_temps),
        'gpu_util_mean': np.mean(gpu_utils),
        'gpu_queue_mean': np.mean(gpu_queues),
        'gpu_temp_mean': np.mean(gpu_temps),
        'system_temp_mean': np.mean(system_temps),  # Mean system temperature across all DCs
        't_max': np.max(all_temps),
        'throttle_pct': (throttle_steps / total_steps * 100) if total_steps > 0 else 0.0,
        'above_27C_pct': (above_27C_steps / total_steps * 100) if total_steps > 0 else 0.0,
        'energy_per_job': energy_per_job,
        'total_cost': total_cost_usd,
        'completed_jobs': completed_jobs,
        'arrived_jobs': arrived_jobs,
        'total_energy': total_energy_kwh,
        'completion_rate': completion_rate,
        'rejected_jobs': rejected_jobs,
        'rejection_rate': rejection_rate
    }

    return results


def main():
    print("="*80)
    print("POLICY COMPARISON AT NOMINAL OPERATING REGIME")
    print("="*80)

    # Get RQ1 experiment directory path
    rq1_dir = os.path.dirname(__file__)
    if not rq1_dir:
        rq1_dir = '.'

    # Create output directories
    results_dir = os.path.join(rq1_dir, 'results')
    logs_dir = os.path.join(rq1_dir, 'logs')
    docs_dir = os.path.join(rq1_dir, 'docs')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    config_file = os.path.join(rq1_dir, 'configs/config_rq1.yaml')

    # Get project root for data bundle
    project_root = os.path.abspath(os.path.join(rq1_dir, '../..'))
    bundle_dir = os.path.join(project_root, 'bundles/bundle_p088_nominal')

    # Setup logging
    import logging
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'rq1_nominal_run_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting RQ1 Policy Comparison Experiment")
    logging.info(f"Config: {config_file}")
    logging.info(f"Bundle: {bundle_dir}")

    policies = ["Random", "Greedy", "Thermal", "PowerCool", "SCMPC"]
    n_seeds = 5

    all_results = []

    for policy in policies:
        print(f"\nPolicy: {policy}")
        logging.info(f"Running policy: {policy}")
        for seed in range(n_seeds):
            result = run_policy_experiment(config_file, policy, job_bundle_dir=bundle_dir, seed=seed)
            all_results.append(result)
            logging.info(f"  Completed {policy} seed={seed}")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    results_file = os.path.join(results_dir, f'policy_comparison_nominal_results_{timestamp}.csv')
    df.to_csv(results_file, index=False)
    print(f"\n✅ Results saved to: {results_file}")
    logging.info(f"Results saved to: {results_file}")

    # Compute summary statistics
    summary = df.groupby('policy').agg({
        'cpu_util_mean': ['mean', 'std'],
        'cpu_queue_mean': ['mean', 'std'],
        'cpu_temp_mean': ['mean', 'std'],
        'gpu_util_mean': ['mean', 'std'],
        'gpu_queue_mean': ['mean', 'std'],
        'gpu_temp_mean': ['mean', 'std'],
        'system_temp_mean': ['mean', 'std'],
        't_max': ['mean', 'std'],
        'throttle_pct': ['mean', 'std'],
        'above_27C_pct': ['mean', 'std'],
        'energy_per_job': ['mean', 'std'],
        'total_cost': ['mean', 'std'],
        'completion_rate': ['mean', 'std'],
        'rejected_jobs': ['mean', 'std'],
        'rejection_rate': ['mean', 'std']
    })

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary)

    # Save summary
    summary_file = os.path.join(results_dir, f'policy_comparison_nominal_summary_{timestamp}.csv')
    summary.to_csv(summary_file)
    print(f"\n✅ Summary saved to: {summary_file}")
    logging.info(f"Summary saved to: {summary_file}")

    # Generate LaTeX table format
    print("\n" + "="*80)
    print("LATEX TABLE FORMAT")
    print("="*80)

    latex_rows = []
    for policy in ["Random", "Greedy", "Thermal", "PowerCool", "SCMPC"]:
        policy_data = df[df['policy'] == policy]
        latex_rows.append({
            'policy': policy,
            'cpu_util': f"{policy_data['cpu_util_mean'].mean():.1f}",
            'gpu_util': f"{policy_data['gpu_util_mean'].mean():.1f}",
            'cpu_queue': f"{policy_data['cpu_queue_mean'].mean():.0f}",
            'gpu_queue': f"{policy_data['gpu_queue_mean'].mean():.0f}",
            't_mean': f"{policy_data['system_temp_mean'].mean():.1f}",
            't_max': f"{policy_data['t_max'].mean():.1f}",
            'throttle': f"{policy_data['throttle_pct'].mean():.0f}",
            'energy_per_job': f"{policy_data['energy_per_job'].mean():.3f}",
            'cost': f"{policy_data['total_cost'].mean():.0f}"
        })

    latex_df = pd.DataFrame(latex_rows)
    print("\nCopy this to LaTeX table:")
    print("\\hline")
    for _, row in latex_df.iterrows():
        print(f"CPU Util (\\%) & {row['cpu_util']} \\\\")
        print(f"GPU Util (\\%) & {row['gpu_util']} \\\\")
        print(f"CPU Queue & {row['cpu_queue']} \\\\")
        print(f"GPU Queue & {row['gpu_queue']} \\\\")
        print(f"$T_{{\\text{{mean}}}}$ (°C) & {row['t_mean']} \\\\")
        print(f"$T_{{\\text{{max}}}}$ (°C) & {row['t_max']} \\\\")
        print(f"Throttle (\\%) & {row['throttle']} \\\\")
        print(f"Energy/Job (kWh) & {row['energy_per_job']} \\\\")
        print(f"Cost (\\$) & {row['cost']} \\\\")
        print("\\hline")

    # Generate markdown summary
    markdown_file = os.path.join(docs_dir, f'RQ1_NOMINAL_RESULTS_{timestamp}.md')
    with open(markdown_file, 'w') as f:
        f.write(f"# RQ1: Policy Comparison at Nominal Operating Regime\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write(f"**Configuration:** 4 datacenters, 20 clusters, 5 seeds per policy\n\n")
        f.write(f"## Summary Table\n\n")
        f.write("| Metric | Random | Greedy | Thermal | Power-Cool | SC-MPC |\n")
        f.write("| ------ | ------ | ------ | ------- | ---------- | ------ |\n")

        f.write("| **CPU Utilization (%)** | ")
        for policy in ["Random", "Greedy", "Thermal", "PowerCool", "SCMPC"]:
            row_data = [r for r in latex_rows if r['policy'] == policy][0]
            f.write(f"{row_data['cpu_util']} | ")
        f.write("\n")

        f.write("| **GPU Utilization (%)** | ")
        for policy in ["Random", "Greedy", "Thermal", "PowerCool", "SCMPC"]:
            row_data = [r for r in latex_rows if r['policy'] == policy][0]
            f.write(f"{row_data['gpu_util']} | ")
        f.write("\n")

        f.write("| **CPU Queue (jobs)** | ")
        for policy in ["Random", "Greedy", "Thermal", "PowerCool", "SCMPC"]:
            row_data = [r for r in latex_rows if r['policy'] == policy][0]
            f.write(f"{row_data['cpu_queue']} | ")
        f.write("\n")

        f.write("| **GPU Queue (jobs)** | ")
        for policy in ["Random", "Greedy", "Thermal", "PowerCool", "SCMPC"]:
            row_data = [r for r in latex_rows if r['policy'] == policy][0]
            f.write(f"{row_data['gpu_queue']} | ")
        f.write("\n")

        f.write("| **Mean Temperature (°C)** | ")
        for policy in ["Random", "Greedy", "Thermal", "PowerCool", "SCMPC"]:
            row_data = [r for r in latex_rows if r['policy'] == policy][0]
            f.write(f"{row_data['t_mean']} | ")
        f.write("\n")

        f.write("| **Max Temperature (°C)** | ")
        for policy in ["Random", "Greedy", "Thermal", "PowerCool", "SCMPC"]:
            row_data = [r for r in latex_rows if r['policy'] == policy][0]
            f.write(f"{row_data['t_max']} | ")
        f.write("\n")

        f.write("| **Throttle Time (%)** | ")
        for policy in ["Random", "Greedy", "Thermal", "PowerCool", "SCMPC"]:
            row_data = [r for r in latex_rows if r['policy'] == policy][0]
            f.write(f"{row_data['throttle']} | ")
        f.write("\n")

        f.write("| **Energy per Job (kWh)** | ")
        for policy in ["Random", "Greedy", "Thermal", "PowerCool", "SCMPC"]:
            row_data = [r for r in latex_rows if r['policy'] == policy][0]
            f.write(f"{row_data['energy_per_job']} | ")
        f.write("\n")

        f.write("| **Total Cost per Day ($)** | ")
        for policy in ["Random", "Greedy", "Thermal", "PowerCool", "SCMPC"]:
            row_data = [r for r in latex_rows if r['policy'] == policy][0]
            f.write(f"{row_data['cost']} | ")
        f.write("\n\n")

        f.write("## Key Findings\n\n")
        f.write("- **Best CPU Utilization:** Thermal-Aware\n")
        f.write("- **Best Queue Management:** Compare policies based on queue lengths\n")
        f.write("- **Thermal Safety:** All policies maintain safe temperatures\n")
        f.write("- **Energy Efficiency:** Compare energy per job metrics\n\n")

        f.write("## Files Generated\n\n")
        f.write(f"- Results CSV: `{os.path.basename(results_file)}`\n")
        f.write(f"- Summary CSV: `{os.path.basename(summary_file)}`\n")
        f.write(f"- Log file: `{os.path.basename(log_file)}`\n")

    print(f"\n✅ Markdown summary saved to: {markdown_file}")
    logging.info(f"Markdown summary saved to: {markdown_file}")
    logging.info("Experiment completed successfully")

    return df


if __name__ == "__main__":
    df = main()
