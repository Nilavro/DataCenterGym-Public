"""
Simulation using Real DataCenterGym Environment with Aligned Configuration

This script runs simulations with the real DataCenterGym environment using
R and C values aligned with Table {tab:experimental_config} from the paper:
- DC0 (Seattle):  R = 0.008 °C/W, C = 600 kJ/°C
- DC1 (Phoenix):  R = 0.012 °C/W, C = 450 kJ/°C
- DC2 (Chicago):  R = 0.010 °C/W, C = 550 kJ/°C
- DC3 (Dallas):   R = 0.011 °C/W, C = 480 kJ/°C
"""

import yaml
import pandas as pd
import numpy as np
import pickle
import os
from datacenter_env import DataCenterEnv
from policies import RandomBaseline, GreedyCapacity, GreedyCapacityThermalAware, PowerCoolingAware

def load_environment(config_path, bundle_dir, seed=0):
    """
    Load and initialize the DataCenterGym environment.

    Args:
        config_path: Path to YAML configuration file
        bundle_dir: Path to processed data bundle
        seed: Random seed for reproducibility

    Returns:
        env: Initialized DataCenterEnv instance
        config: Configuration dictionary
    """
    print(f"Loading configuration from: {config_path}")

    # Load config
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Load job trace
    cluster_map_path = os.path.join(bundle_dir, "cluster_map.pkl")
    job_trace_path = os.path.join(bundle_dir, "job_trace.parquet")
    usage_trace_path = os.path.join(bundle_dir, "usage_trace.parquet")

    print(f"Loading data from: {bundle_dir}")
    with open(cluster_map_path, "rb") as f:
        cluster_map = pickle.load(f)

    job_trace_df = pd.read_parquet(job_trace_path)
    job_trace = {}
    for row in job_trace_df.itertuples():
        job = {
            "id": row.id,
            "r": row.r,
            "d": row.d,
            "v": row.v,
            "tau": getattr(row, 'tau', 'cpu')
        }
        time_step = int(row.time)
        if time_step not in job_trace:
            job_trace[time_step] = []
        if len(job_trace[time_step]) < 200:
            job_trace[time_step].append(job)

    usage_trace_df = pd.read_parquet(usage_trace_path)

    # Build config
    cluster_params = []
    for cluster_cfg in yaml_config['clusters']:
        cluster_params.append({
            'type': cluster_cfg.get('type', 'cpu'),
            'alpha': cluster_cfg.get('alpha', 0.8),
            'phi': cluster_cfg.get('phi', 0.2),
            'c_max': cluster_cfg.get('c_max', 200)
        })

    datacenter_params = []
    for dc_cfg in yaml_config['datacenters']:
        datacenter_params.append({
            'name': dc_cfg.get('name', 'DC'),
            'location': dc_cfg.get('location', 'Unknown'),
            'R': dc_cfg.get('R', 0.01),
            'C': dc_cfg.get('C', 500000.0),
            'theta_max': dc_cfg.get('theta_max', 35),
            'theta_init': dc_cfg.get('theta_init', 20),
            'cooling_max': dc_cfg.get('cooling_max', 30000),
            'T_target': dc_cfg.get('T_target', 22),
            'controller_type': dc_cfg.get('controller_type', 'proportional'),
            'controller_params': dc_cfg.get('controller_params', {}),
            'climate': dc_cfg.get('climate', {})
        })

    # Print R and C values to verify alignment
    print("\n" + "="*80)
    print("DATACENTER THERMAL PARAMETERS (Aligned with Table {tab:experimental_config})")
    print("="*80)
    for i, dc in enumerate(datacenter_params):
        print(f"{dc['location']:<12} | R = {dc['R']:.4f} °C/W | C = {dc['C']:>10.1f} J/°C ({dc['C']/1000:.1f} kJ/°C)")
    print("="*80 + "\n")

    config = {
        'num_clusters': yaml_config['num_clusters'],
        'num_datacenters': yaml_config['num_datacenters'],
        'episode_length': yaml_config.get('episode_length', 288),
        'cluster_params': cluster_params,
        'datacenter_params': datacenter_params,
        'cluster_to_dc': yaml_config['cluster_to_dc'],
        'lambda_weights': yaml_config.get('lambda_weights', {}),
        'enforce_affinity': yaml_config.get('enforce_affinity', True),
        'pricing': yaml_config.get('pricing', {})
    }

    # Create environment
    np.random.seed(seed)
    env = DataCenterEnv(cluster_map, job_trace, usage_trace_df, None, config)

    return env, config


def run_simulation(env, policy, policy_name, seed=0, verbose=True):
    """
    Run a single simulation episode with the given policy.

    Args:
        env: DataCenterEnv instance
        policy: Scheduling policy instance
        policy_name: Name of the policy
        seed: Random seed
        verbose: Whether to print progress

    Returns:
        results: Dictionary with simulation results
    """
    print(f"\n{'='*80}")
    print(f"Running Simulation: {policy_name} (seed={seed})")
    print(f"{'='*80}")

    obs, info = env.reset()
    done = False
    step = 0

    # Tracking metrics
    temperatures = {dc_id: [] for dc_id in env.datacenters.keys()}
    cooling_powers = {dc_id: [] for dc_id in env.datacenters.keys()}
    utilizations = {dc_id: [] for dc_id in env.datacenters.keys()}
    queue_lengths = {dc_id: [] for dc_id in env.datacenters.keys()}

    total_reward = 0.0

    while not done:
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward

        # Collect metrics
        for dc_id, dc in env.datacenters.items():
            temperatures[dc_id].append(dc['theta'])
            cooling_powers[dc_id].append(dc['cooling_power'])

            # Get clusters belonging to this datacenter
            cluster_indices = [i for i, d in env.cluster_to_dc.items() if d == dc_id]
            dc_util = np.mean([env.clusters[i]['u'] for i in cluster_indices]) if cluster_indices else 0
            dc_queue = np.mean([env.clusters[i]['q'] for i in cluster_indices]) if cluster_indices else 0

            utilizations[dc_id].append(dc_util)
            queue_lengths[dc_id].append(dc_queue)

        step += 1

        if verbose and step % 50 == 0:
            print(f"  Step {step}/288 | Reward: {reward:.2f} | Energy: {info['step_energy_kwh']:.2f} kWh | Cost: ${info['step_cost_usd']:.2f}")

    # Compute statistics
    results = {
        'policy': policy_name,
        'seed': seed,
        'total_reward': total_reward,
        'total_energy_kwh': env.total_energy_kwh,
        'total_cost_usd': env.total_cost_usd
    }

    # Per-datacenter statistics
    for dc_id in env.datacenters.keys():
        location = env.config['datacenter_params'][dc_id]['location']
        results[f'{location}_temp_mean'] = np.mean(temperatures[dc_id])
        results[f'{location}_temp_max'] = np.max(temperatures[dc_id])
        results[f'{location}_temp_std'] = np.std(temperatures[dc_id])
        results[f'{location}_cooling_mean'] = np.mean(cooling_powers[dc_id])
        results[f'{location}_cooling_max'] = np.max(cooling_powers[dc_id])
        results[f'{location}_util_mean'] = np.mean(utilizations[dc_id])
        results[f'{location}_queue_mean'] = np.mean(queue_lengths[dc_id])

    print(f"\n✅ Simulation Complete!")
    print(f"   Total Energy: {env.total_energy_kwh:.2f} kWh")
    print(f"   Total Cost: ${env.total_cost_usd:.2f}")
    print(f"   Total Reward: {total_reward:.2f}")

    return results


def run_monte_carlo_experiment(config_path, bundle_dir, policies_config, n_seeds=10, output_file='simulation_results.csv'):
    """
    Run Monte Carlo experiments with multiple policies and seeds.

    Args:
        config_path: Path to configuration YAML
        bundle_dir: Path to data bundle
        policies_config: List of (policy_name, policy_class, policy_kwargs) tuples
        n_seeds: Number of random seeds to run
        output_file: Path to save results CSV
    """
    all_results = []

    for seed in range(n_seeds):
        print(f"\n\n{'#'*80}")
        print(f"# SEED {seed}/{n_seeds}")
        print(f"{'#'*80}")

        for policy_name, policy_class, policy_kwargs in policies_config:
            # Load fresh environment for each run
            env, config = load_environment(config_path, bundle_dir, seed=seed)

            # Create policy
            policy = policy_class(env, **policy_kwargs)

            # Run simulation
            results = run_simulation(env, policy, policy_name, seed=seed, verbose=True)
            all_results.append(results)

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"\n\n{'='*80}")
    print(f"✅ All simulations complete! Results saved to: {output_file}")
    print(f"{'='*80}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Mean ± Std across seeds)")
    print("="*80)

    for policy_name in df['policy'].unique():
        policy_df = df[df['policy'] == policy_name]
        print(f"\n{policy_name}:")
        print(f"  Energy: {policy_df['total_energy_kwh'].mean():.2f} ± {policy_df['total_energy_kwh'].std():.2f} kWh")
        print(f"  Cost: ${policy_df['total_cost_usd'].mean():.2f} ± ${policy_df['total_cost_usd'].std():.2f}")
        print(f"  Reward: {policy_df['total_reward'].mean():.2f} ± {policy_df['total_reward'].std():.2f}")

        # Datacenter temperatures
        for location in ['Seattle', 'Phoenix', 'Chicago', 'Dallas']:
            temp_col = f'{location}_temp_mean'
            if temp_col in policy_df.columns:
                print(f"  {location} Temp: {policy_df[temp_col].mean():.2f} ± {policy_df[temp_col].std():.2f} °C")

    return df


def main():
    """Main execution."""

    # Configuration
    config_path = 'configs/config_aligned_experimental.yaml'
    bundle_dir = 'processed_bundle_nominal_65'

    # Define policies to evaluate
    policies_config = [
        ('Random', RandomBaseline, {'target_temp': 22.0}),
        ('GreedyCapacity', GreedyCapacity, {'target_temp': 22.0}),
        ('ThermalAware', GreedyCapacityThermalAware, {'target_temp': 22.0, 'thermal_weight': 0.4}),
        ('PowerCooling', PowerCoolingAware, {'target_temp': 22.0, 'cooling_weight': 1.0})
    ]

    # Run Monte Carlo experiments
    print("\n" + "="*80)
    print("DATACENTER GYM SIMULATION WITH ALIGNED CONFIGURATION")
    print("="*80)
    print(f"Configuration: {config_path}")
    print(f"Data Bundle: {bundle_dir}")
    print(f"Policies: {[p[0] for p in policies_config]}")
    print(f"Number of seeds: 10")
    print("="*80 + "\n")

    results_df = run_monte_carlo_experiment(
        config_path=config_path,
        bundle_dir=bundle_dir,
        policies_config=policies_config,
        n_seeds=10,
        output_file='datacenter_simulation_results_aligned.csv'
    )

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
