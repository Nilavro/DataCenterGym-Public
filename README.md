# DataCenterGym: Physics-Grounded Datacenter Scheduling Simulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Gymnasium-compatible environment for multi-objective datacenter scheduling with thermal management, cooling control, and geo-distributed optimization.

## 📄 Paper

**DataCenterGym: A Physics-Grounded Simulator for Multi-Objective Data Center Scheduling**

Nilavra Pathak (Expedia Group), Samadrita Biswas (Curia Global), Nirmalya Roy (UMBC)

Accepted at IEEE SMARTCOMP 2026

## 🎯 Key Features

- **Physics-Based Thermal Dynamics**: RC thermal model with PID cooling control
- **Realistic Workloads**: Alibaba 2018 cluster trace with CPU/GPU affinity
- **Multiple Baselines**: 6 scheduling policies from random to hierarchical MPC
- **Gymnasium Compatible**: Standard RL environment interface
- **Reproducible Results**: Complete experiment configurations for paper replication

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/[ORG]/DataCenterGym.git
cd DataCenterGym
pip install -r requirements.txt
```

### Run Your First Simulation

```python
from datacenter_env import DataCenterEnv
from policies.all_policies import GreedyCapacityThermalAware
import yaml
import pickle
import pandas as pd

# Load configuration
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

# Load workload
with open('bundles/bundle_p088_nominal/cluster_map.pkl', 'rb') as f:
    cluster_map = pickle.load(f)
job_trace = pd.read_parquet('bundles/bundle_p088_nominal/job_trace.parquet')
usage_trace = pd.read_parquet('bundles/bundle_p088_nominal/usage_trace.parquet')

# Initialize environment
env = DataCenterEnv(cluster_map, job_trace, usage_trace, None, config)

# Create policy
policy = GreedyCapacityThermalAware(env, target_temp=22.0, thermal_weight=0.4)

# Run episode
obs, info = env.reset()
done = False
total_energy = 0

while not done:
    action = policy.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Episode completed!")
print(f"Total Energy: {env.total_energy_kwh:.2f} kWh")
print(f"Total Cost: ${env.total_cost_usd:.2f}")
print(f"Max Temperature: {max(env.history['temperature_mean']):.1f}°C")
```

## 📊 Reproducing Paper Results

### RQ1: Policy Comparison (Table 1)

```bash
cd experiments/RQ1
python run_policy_comparison_nominal.py
```

**Expected Output**: Comparative metrics for 6 policies under nominal workload (200 jobs/step, 60-70% utilization)

### RQ2: Workload Sensitivity (Figure 3)

```bash
cd experiments/RQ2
python run_rq2_hmpc_lambda_sweep.py
python run_rq2_baseline_lambda_sweep.py
```

**Expected Output**: Performance across arrival rate scaling factors (0.5x to 3.0x)

### RQ3: Thermal Management (Figure 4)

```bash
cd experiments/RQ3
python final_thermal_comparison_paper_config.py
```

**Expected Output**: Temperature distributions under moderate/extreme thermal stress

## 🏗️ Architecture

### Environment (`datacenter_env/`)

**`datacenter_env.py`**: Main Gymnasium environment
- Observation space: Per-cluster state (capacity, queue, utilization) + per-datacenter temperature
- Action space: Job assignments + temperature setpoints
- Reward: Multi-objective (queue length, temperature, energy, cost)

**`dynamics.py`**: Thermal RC model + HVAC controllers
- RC thermal dynamics: `C × dθ/dt = Q_gen - (θ - T_amb)/R - Q_hvac`
- PID cooling control with anti-windup
- Soft thermal throttling mechanism

**`weather.py`**: Climate profiles for 4 datacenters (Seattle, Phoenix, Chicago, Dallas)

### Policies (`policies/`)

**`all_policies.py`**: Baseline heuristics
- `RandomBaseline`: Uniform random assignment
- `GreedyCapacity`: Minimize utilization
- `GreedyCapacityThermalAware`: Balance capacity + temperature
- `PowerCoolingAware`: Minimize energy cost

**`all_policies_hierarchical_mpc.py`**: Hierarchical MPC (H-MPC)
- Stage 1: Datacenter-level admission control + thermal setpoints
- Stage 2: Cluster-level job allocation
- Complexity: O(D³H³) + D·O((CJH/D²)³)

### System Configuration

**Topology**: 4 datacenters × 20 heterogeneous clusters

**Datacenters**:
- Seattle: Cool (10±5°C), 0.68MW cooling, $0.06-0.08/kWh
- Phoenix: Hot (38±12°C), 1.22MW cooling, $0.14-0.22/kWh
- Chicago: Temperate (16±10°C), 0.30MW cooling, $0.09-0.13/kWh
- Dallas: Warm (30±11°C), 1.97MW cooling, $0.11-0.19/kWh

**Episode**: 24 hours @ 5-minute timesteps (288 steps)

## 📈 Performance Metrics

| Dimension | Metrics |
|-----------|---------|
| **QoS** | CPU/GPU Utilization, Queue Length |
| **Thermal** | Mean/Max Temperature, Throttling % |
| **Energy** | Total kWh, kWh/Job, Operational Cost ($) |

## 🔧 Configuration

Key parameters in `configs/config.yaml`:

```yaml
# Thermal properties (per datacenter)
R: 0.003-0.005              # °C/W - thermal resistance
C: 520000-700000            # J/°C - thermal capacitance
theta_max: 27-35            # °C - safe operating zone
cooling_max: 70000-100000   # W - max HVAC power

# PID controller
K_p: 4000-7000              # W/°C
K_i: 80-150                 # W/(°C·s)
K_d: 800-1500               # W·s/°C

# Heat generation
alpha_cpu: 0.3-0.8          # W/CU
alpha_gpu: 4.0-9.0          # W/CU

# Throttling
theta_soft: 32              # °C - throttling onset
g_min: 0.2-0.7              # Minimum capacity fraction
```

## 📝 Citation

```bibtex
@inproceedings{datacentergym2025,
  title={DataCenterGym: A Physics-Grounded Simulator for Multi-Objective Data Center Scheduling},
  author={Pathak, Nilavra and Biswas, Samadrita and Roy, Nirmalya},
  booktitle={IEEE International Conference on Smart Computing (SMARTCOMP)},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact:
- Nilavra Pathak: npathak@expediagroup.com

## 🙏 Acknowledgments

 
This work has been partially supported by NSF CAREER Grant \#1750936, NSF CNS EAGER Grant \#2233879, NSF IIS Grant \#2509680,  NSF I-Corps Grant \# 2502886, U.S. Army Grants \#W911NF2120076 & \#W911NF2410367, and ONR Grant \#N00014-23-1-2119. 
---

**Workload Data**: Alibaba 2018 cluster trace available at [https://github.com/alibaba/clusterdata](https://github.com/alibaba/clusterdata)
