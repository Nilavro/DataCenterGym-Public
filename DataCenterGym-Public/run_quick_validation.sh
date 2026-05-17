#!/bin/bash
# Quick validation script to verify DataCenterGym setup and basic functionality
# Runtime: ~5 minutes

set -e

echo "=========================================="
echo "DataCenterGym Quick Validation"
echo "=========================================="
echo ""

# Check we're in the right directory
if [ ! -f "datacenter_env/datacenter_env.py" ]; then
    echo "❌ Error: Run this script from DataCenterGym-Public directory"
    exit 1
fi

echo "✓ Directory check passed"

# Test imports
echo ""
echo "Testing imports..."
python3 -c "
from datacenter_env import DataCenterEnv
from policies.all_policies import GreedyCapacity, GreedyCapacityThermalAware, RandomBaseline
print('✓ Core imports successful')
" || { echo "❌ Import failed. Run: pip install gymnasium numpy pandas pyyaml"; exit 1; }

# Run minimal simulation test
echo ""
echo "Running minimal simulation (20 steps)..."
python3 << 'PYEOF'
import sys
from load_config_helper import load_config, load_workload
from datacenter_env import DataCenterEnv
from policies.all_policies import GreedyCapacity

# Load config using helper
config = load_config('configs/config.yaml')

# Load workload using helper
cluster_map, job_trace, usage_trace = load_workload('bundles/bundle_p088_nominal')

# Run short episode
env = DataCenterEnv(cluster_map, job_trace, usage_trace, None, config)
policy = GreedyCapacity(env)

obs, _ = env.reset()
for step in range(20):
    action = policy.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# Validate basic metrics
assert env.total_energy_kwh > 0, "Energy should be positive"
assert env.total_cost_usd > 0, "Cost should be positive"
assert env.t > 0, "Timesteps should advance"

# obs is flat array: [theta, p, c, q, T_amb] per cluster + [T_target] per DC
# First element is temperature of first cluster's datacenter
temp = obs[0]
assert 15 < temp < 40, f"Temperature {temp}°C out of reasonable range"

print(f"✓ Simulation successful!")
print(f"  Steps: {env.t}")
print(f"  Energy: {env.total_energy_kwh:.2f} kWh")
print(f"  Cost: ${env.total_cost_usd:.2f}")
print(f"  Temperature: {temp:.1f}°C")
PYEOF

if [ $? -eq 0 ]; then
    echo "✓ Validation passed"
else
    echo "❌ Simulation failed"
    exit 1
fi

# Test multiple policies
echo ""
echo "Testing multiple policies (10 steps each)..."
python3 << 'PYEOF'
from load_config_helper import load_config, load_workload
from datacenter_env import DataCenterEnv
from policies.all_policies import (
    GreedyCapacity,
    GreedyCapacityThermalAware,
    RandomBaseline,
    PowerCoolingAware
)

# Load using helper
config = load_config('configs/config.yaml')
cluster_map, job_trace, usage_trace = load_workload('bundles/bundle_p088_nominal')

policies = [
    ("Random", RandomBaseline),
    ("Greedy", GreedyCapacity),
    ("ThermalAware", lambda env: GreedyCapacityThermalAware(env, target_temp=22.0, thermal_weight=0.4)),
    ("PowerCool", PowerCoolingAware)
]

results = []
for name, policy_class in policies:
    env = DataCenterEnv(cluster_map, job_trace, usage_trace, None, config)
    policy = policy_class(env)

    obs, _ = env.reset()
    for step in range(10):
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results.append({
        'policy': name,
        'energy': env.total_energy_kwh,
        'cost': env.total_cost_usd
    })
    print(f"  {name:15s}: {env.total_energy_kwh:6.2f} kWh, ${env.total_cost_usd:7.2f}")

# Validate ordering (not strict, just sanity check)
energies = [r['energy'] for r in results]
assert max(energies) / min(energies) < 2.0, "Energy variance too high (>2x)"
print("✓ All policies executed successfully")
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ Policy test failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ All validations passed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - Full replication: See REPLICATION.md"
echo "  - RQ1 (30 min): cd experiments/RQ1 && python3 run_policy_comparison_nominal.py"
echo "  - Quick tutorial: See QUICKSTART.md"
echo ""
