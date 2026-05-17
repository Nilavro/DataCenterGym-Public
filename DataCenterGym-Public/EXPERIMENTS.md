# Experiment Reproduction Guide

Complete instructions to reproduce all paper results.

## Setup

Ensure you've installed dependencies:
```bash
pip install -r requirements.txt
```

## RQ1: Policy Comparison

**Research Question**: How do different scheduling policies perform under nominal conditions?

**Command**:
```bash
cd experiments/RQ1
python run_policy_comparison_nominal.py
```

**Configuration**: `configs/config.yaml`
- Workload: 200 jobs/step (60-70% utilization)
- Episode: 24 hours (288 timesteps)
- Seeds: 5 independent runs

**Output**: `results/rq1_comparison.csv`

**Expected Results** (Table 1 in paper):
| Policy | CPU Queue | GPU Queue | kWh/Job | Cost ($) |
|--------|-----------|-----------|---------|----------|
| H-MPC | 324±15 | 449±22 | 2.20±0.05 | 14,424±180 |
| PowerCool | 356±18 | 502±25 | 2.26±0.06 | 15,109±220 |
| Greedy | 338±17 | 511±28 | 2.34±0.07 | 15,880±250 |

**Runtime**: ~30 minutes on 8-core CPU

---

## RQ2: Workload Sensitivity

**Research Question**: How robust are policies under varying arrival rates?

**Commands**:
```bash
cd experiments/RQ2
# Run H-MPC sweep
python run_rq2_hmpc_lambda_sweep.py

# Run baseline sweeps
python run_rq2_baseline_lambda_sweep.py
```

**Configuration**: Arrival rate scaling λ ∈ {0.5, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0}

**Output**: `results/rq2_lambda_sweep_*.csv`

**Expected Observation**: H-MPC maintains 60-70% target utilization across all λ, while Greedy/PowerCool saturate near λ≈1.6

**Runtime**: ~2 hours (parallelizable across λ values)

---

## RQ3: Thermal Management

**Research Question**: Does HVAC maintain safe operating zone under thermal stress?

**Command**:
```bash
cd experiments/RQ3
python final_thermal_comparison_paper_config.py
```

**Scenarios**:
1. Moderate heat: Phoenix ambient 38±12°C
2. Extreme heat: Phoenix ambient 48±12°C (+10°C)

**Output**: Temperature distribution plots in `figures/rq3_thermal_*.png`

**Expected Result**: H-MPC maintains θ<27°C under moderate stress, requires throttling under extreme stress but remains stable.

**Runtime**: ~45 minutes

---

## Troubleshooting

### "Module not found" errors
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Solver errors (CVXPY)
Install MOSEK (free academic license):
```bash
pip install mosek
```

### Memory issues
Reduce episode length in config:
```yaml
episode_length: 144  # 12 hours instead of 24
```

---

## Custom Experiments

To run your own configuration:

```python
from experiments.common.simulate_real_datacenter_env import simulate_episode
import yaml

# Load custom config
with open('configs/my_config.yaml') as f:
    config = yaml.safe_load(f)

# Run simulation
results = simulate_episode(
    config=config,
    policy_name='GreedyCapacityThermalAware',
    bundle_path='bundles/bundle_p088_nominal',
    seed=42
)

print(f"Energy: {results['total_energy_kwh']:.2f} kWh")
print(f"Cost: ${results['total_cost_usd']:.2f}")
```
