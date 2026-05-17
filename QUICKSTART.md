# DataCenterGym Quick Start Guide

Get running in 5 minutes!

## Step 1: Install (2 minutes)

```bash
git clone https://github.com/[ORG]/DataCenterGym.git
cd DataCenterGym
pip install -r requirements.txt
```

## Step 2: Verify Installation (1 minute)

```bash
python -c "from datacenter_env import DataCenterEnv; print('✓ Environment loaded successfully!')"
```

## Step 3: Run First Simulation (2 minutes)

Create `test_sim.py`:

```python
from datacenter_env import DataCenterEnv
from policies.all_policies import GreedyCapacity
import yaml, pickle, pandas as pd

# Load config and data
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)
with open('bundles/bundle_p088_nominal/cluster_map.pkl', 'rb') as f:
    cluster_map = pickle.load(f)
job_trace = pd.read_parquet('bundles/bundle_p088_nominal/job_trace.parquet')
usage_trace = pd.read_parquet('bundles/bundle_p088_nominal/usage_trace.parquet')

# Run
env = DataCenterEnv(cluster_map, job_trace, usage_trace, None, config)
policy = GreedyCapacity(env)
obs, _ = env.reset()

for step in range(50):  # 50 steps ≈ 4 hours
    action = policy.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 10 == 0:
        print(f"Step {step}: Energy={env.total_energy_kwh:.1f} kWh, "
              f"Temp={obs['datacenter_state'][0]['temperature']:.1f}°C")
    if terminated or truncated:
        break

print(f"\n✓ Simulation complete! Total cost: ${env.total_cost_usd:.2f}")
```

Run:
```bash
python test_sim.py
```

## What's Next?

- **Compare Policies**: Run `experiments/RQ1/run_policy_comparison_nominal.py`
- **Stress Test**: Try `configs/config_thermal_extreme_paper.yaml`
- **Custom Policy**: Implement your own scheduler in `policies/`

See full `README.md` for details!
