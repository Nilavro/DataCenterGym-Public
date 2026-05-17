"""
Helper functions to load and convert DataCenterGym configurations.
Handles conversion between YAML format and environment format.
"""

import yaml


def load_config(config_path):
    """
    Load YAML config and convert to DataCenterEnv-compatible format.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        config: Dictionary compatible with DataCenterEnv
    """
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Convert cluster configs
    cluster_params = []
    for cluster_cfg in yaml_config['clusters']:
        cluster_params.append({
            'type': cluster_cfg.get('type', 'cpu'),
            'alpha': cluster_cfg.get('alpha', 0.8),
            'phi': cluster_cfg.get('phi', 0.2),
            'c_max': cluster_cfg.get('c_max', 200)
        })

    # Convert datacenter configs (list -> list with proper keys)
    datacenter_params = []
    for dc_cfg in yaml_config['datacenters']:
        # Extract thermal throttling config if exists
        throttling_config = dc_cfg.get('thermal_throttling', {})

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
            'climate': dc_cfg.get('climate', {}),
            'thermal_throttling': {
                'enabled': throttling_config.get('enabled', False),
                'theta_soft': throttling_config.get('theta_soft', 32.0),
                'g_min': throttling_config.get('g_min', 0.5)
            }
        })

    # Build environment config
    config = {
        'num_clusters': yaml_config['num_clusters'],
        'num_datacenters': yaml_config['num_datacenters'],
        'episode_length': yaml_config.get('episode_length', 288),
        'time_step_minutes': yaml_config.get('time_step_minutes', 5),
        'cluster_params': cluster_params,
        'datacenter_params': datacenter_params,
        'cluster_to_dc': yaml_config['cluster_to_dc'],
        'lambda_weights': yaml_config.get('lambda_weights', {}),
        'enforce_affinity': yaml_config.get('enforce_affinity', True),
        'pricing': yaml_config.get('pricing', {})
    }

    return config


def load_workload(bundle_dir):
    """
    Load preprocessed workload bundle (cluster_map, job_trace, usage_trace).

    Args:
        bundle_dir: Path to bundle directory

    Returns:
        cluster_map: Dict mapping clusters to datacenters
        job_trace: Dict mapping timesteps to job lists
        usage_trace_df: DataFrame with usage patterns
    """
    import os
    import pickle
    import pandas as pd

    cluster_map_path = os.path.join(bundle_dir, "cluster_map.pkl")
    job_trace_path = os.path.join(bundle_dir, "job_trace.parquet")
    usage_trace_path = os.path.join(bundle_dir, "usage_trace.parquet")

    # Load cluster map
    with open(cluster_map_path, "rb") as f:
        cluster_map = pickle.load(f)

    # Load job trace and convert to dict format
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
        job_trace[time_step].append(job)

    # Load usage trace
    usage_trace_df = pd.read_parquet(usage_trace_path)

    return cluster_map, job_trace, usage_trace_df
