# Paper Results Replication Guide

This guide provides **exact steps** to reproduce all tables and figures from the paper:

**"DataCenterGym: A Physics-Grounded Simulator for Multi-Objective Data Center Scheduling"**

## Prerequisites (5 minutes)

```bash
# Navigate to this directory
cd DataCenterGym-Public

# Install dependencies
pip install gymnasium numpy pandas pyyaml matplotlib seaborn cvxpy casadi

# Verify installation
python3 -c "from datacenter_env import DataCenterEnv; print('✓ Ready!')"
```

---

## Table 1: Policy Comparison Under Nominal Workload (RQ1)

**Paper Location**: Section VI, Table 1

**Expected Results**:

| Policy | CPU Queue | GPU Queue | kWh/Job | Cost ($) | Max Temp (°C) |
|--------|-----------|-----------|---------|----------|---------------|
| H-MPC | 324±15 | 449±22 | 2.20±0.05 | 14,424±180 | 26.3±0.4 |
| SC-MPC | 332±16 | 465±24 | 2.22±0.06 | 14,560±190 | 26.5±0.5 |
| PowerCool | 356±18 | 502±25 | 2.26±0.06 | 15,109±220 | 26.8±0.5 |
| Thermal-Aware | 361±19 | 518±27 | 2.29±0.07 | 15,450±240 | 26.6±0.4 |
| Greedy | 338±17 | 511±28 | 2.34±0.07 | 15,880±250 | 27.2±0.6 |
| Random | 412±25 | 589±35 | 2.41±0.09 | 16,520±310 | 27.8±0.7 |

### Reproduce

**Quick Test (5 minutes, 1 seed)**:
```bash
cd experiments/RQ1
python3 run_policy_comparison_nominal.py --seeds 1 --quick
```

**Full Reproduction (30 minutes, 5 seeds matching paper)**:
```bash
cd experiments/RQ1
python3 run_policy_comparison_nominal.py --seeds 5
```

**Output**: `results/rq1_table1_comparison.csv`

**Configuration Used**:
- Workload: `bundle_p088_nominal` (200 jobs/step, 60-70% target utilization)
- Episode length: 24 hours (288 timesteps @ 5min intervals)
- Monte Carlo: 5 independent random seeds
- Seeds: [42, 123, 456, 789, 1024]

**Validation**: Values should match within ±10% due to stochasticity. Key ranking order (best→worst):
1. H-MPC (lowest queue, lowest energy)
2. SC-MPC (close to H-MPC)
3. PowerCool (moderate energy, higher queue)
4. Greedy (higher energy)
5. Random (worst performance)

---

## Figure 3: Workload Sensitivity Analysis (RQ2)

**Paper Location**: Section VI-B, Figure 3 (Pareto frontiers)

**Expected Observation**: 
- H-MPC maintains 60-70% utilization across all arrival rates
- Greedy and PowerCool saturate (>90% utilization) at λ≈1.6×
- H-MPC shows linear energy scaling, others show knee at saturation

### Reproduce

**Quick Test (10 minutes, 2 arrival rates)**:
```bash
cd experiments/RQ2
python3 run_rq2_hmpc_lambda_sweep.py --lambda-values 1.0 1.6 --seeds 1
python3 run_rq2_baseline_lambda_sweep.py --lambda-values 1.0 1.6 --seeds 1
```

**Full Reproduction (2 hours, 8 arrival rates × 5 seeds)**:
```bash
cd experiments/RQ2

# Run H-MPC across all λ values
python3 run_rq2_hmpc_lambda_sweep.py --lambda-values 0.5 0.8 1.0 1.2 1.6 2.0 2.5 3.0 --seeds 5

# Run baselines
python3 run_rq2_baseline_lambda_sweep.py --lambda-values 0.5 0.8 1.0 1.2 1.6 2.0 2.5 3.0 --seeds 5

# Generate plot
python3 plot_rq2_frontiers.py
```

**Output**: `figures/rq2_pareto_frontier.png`

**Configuration**:
- Base workload: `bundle_p088_nominal`
- Arrival scaling: λ ∈ {0.5, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0}
- Each λ: 5 seeds, 24-hour episodes

**Validation**: 
- At λ=1.0 (nominal): CPU util 60-70%, GPU util 65-75%
- At λ=1.6: Greedy shows >85% utilization, H-MPC stays <75%
- At λ=3.0: Greedy saturated (>95%), queues explode

---

## Figure 4: Thermal Management Under Heat Stress (RQ3)

**Paper Location**: Section VI-C, Figure 4 (Temperature distributions)

**Expected Results**:
- **Moderate Heat (Phoenix 38±12°C)**: All policies maintain θ<30°C, no throttling
- **Extreme Heat (Phoenix 48±12°C)**: H-MPC uses throttling, maintains θ<33°C stable

### Reproduce

**Quick Test (5 minutes, 50 timesteps)**:
```bash
cd experiments/RQ3
python3 final_thermal_comparison_paper_config.py --steps 50 --quick
```

**Full Reproduction (45 minutes, 24 hours)**:
```bash
cd experiments/RQ3
python3 final_thermal_comparison_paper_config.py

# Generate plots
python3 plot_thermal_distributions.py
```

**Output**: 
- `figures/rq3_thermal_distributions_moderate.png`
- `figures/rq3_thermal_distributions_extreme.png`

**Configuration**:
- Scenario 1: `configs/config_thermal_moderate_paper.yaml` (Phoenix 38±12°C)
- Scenario 2: `configs/config_thermal_extreme_paper.yaml` (Phoenix 48±12°C)
- Episode: 24 hours, seeds: 5

**Validation**:
- Moderate: Mean temp 24-28°C, no throttling events
- Extreme: Mean temp 28-32°C, throttling 10-30% of time
- H-MPC tighter temperature bounds than Greedy (±2°C vs ±4°C)

---

## Appendix Figure: Long-Term Thermal Stability

**Paper Location**: Appendix A, 72-hour equilibrium test

**Expected Result**: Temperature converges to equilibrium within 2 hours, remains stable for 72 hours

### Reproduce

```bash
cd experiments/RQ3
python3 run_long_term_thermal_validation.py --duration 72
```

**Output**: `figures/appendix_thermal_equilibrium_72h.png`

**Runtime**: ~6 hours (can run overnight)

---

## Quick Validation Test (Total: 10 minutes)

Run all experiments in "quick mode" to verify setup:

```bash
# From DataCenterGym-Public directory
./run_quick_validation.sh
```

This runs:
- RQ1: 1 seed, 3 policies
- RQ2: 2 arrival rates, 2 policies
- RQ3: 50 timesteps

**Expected Output**:
```
✓ RQ1: H-MPC queue length 320-340 (within range)
✓ RQ2: Utilization scaling correct
✓ RQ3: Temperature stability confirmed
All validations passed! Ready for full replication.
```

---

## Expected Runtimes (8-core CPU, 16GB RAM)

| Experiment | Quick Test | Full Replication |
|------------|------------|------------------|
| RQ1 (Table 1) | 5 min | 30 min |
| RQ2 (Figure 3) | 10 min | 2 hours |
| RQ3 (Figure 4) | 5 min | 45 min |
| **Total** | **20 min** | **3.25 hours** |

Experiments are parallelizable - run multiple λ values or seeds in parallel to reduce wall-clock time.

---

## Troubleshooting

### Results don't match exactly

**Acceptable variance**: ±10% due to Monte Carlo randomness. Key is relative ranking.

**Check**:
1. Using correct seeds: [42, 123, 456, 789, 1024]
2. Using correct bundles: `bundle_p088_nominal` for RQ1/RQ2
3. Episode length: 288 timesteps (24 hours)

### "CVXPY solver failed"

**Solution**: Install better solver (optional but recommended)
```bash
# Academic license (free): https://www.mosek.com/products/academic-licenses/
pip install mosek
```

Default solver (ECOS) works but may be slower for large problems.

### Memory issues

**Symptom**: "MemoryError" or killed process

**Solutions**:
1. Reduce seeds: `--seeds 1` instead of 5
2. Reduce episode length in config: `episode_length: 144` (12 hours)
3. Run experiments sequentially instead of parallel

### Import errors

```bash
# Ensure you're in DataCenterGym-Public directory
cd DataCenterGym-Public
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 -c "from datacenter_env import DataCenterEnv; print('OK')"
```

---

## Configuration Files Reference

All experiments use configs from `configs/` directory:

| Config File | Purpose | Key Settings |
|-------------|---------|--------------|
| `config.yaml` | Base configuration | 4 DCs, 20 clusters, nominal thermal |
| `config_nominal_60.yaml` | RQ1 low load | 160 jobs/step (60% util) |
| `config_nominal_70.yaml` | RQ1 high load | 240 jobs/step (70% util) |
| `config_thermal_moderate_paper.yaml` | RQ3 moderate | Phoenix 38±12°C |
| `config_thermal_extreme_paper.yaml` | RQ3 extreme | Phoenix 48±12°C |

To inspect a config:
```bash
cat configs/config.yaml
```

---

## Workload Bundles Reference

All experiments use preprocessed Alibaba 2018 trace bundles from `bundles/`:

| Bundle | Jobs/Step | Target Utilization | Used In |
|--------|-----------|-------------------|---------|
| `bundle_p088_nominal` | 200 | 65-70% | RQ1, RQ2 baseline |
| `bundle_p07_nominal` | 170 | 60-65% | Sensitivity analysis |
| `bundle_nominal_185jobs` | 185 | 62-68% | Alternative baseline |

Each bundle contains:
- `cluster_map.pkl`: Hardware topology (20 clusters, 4 datacenters)
- `job_trace.parquet`: Job arrivals (resource demand, duration, affinity)
- `usage_trace.parquet`: Historical machine utilization patterns
- `machine_meta.parquet`: Cluster capacities (CPU/GPU CU)

---

## Detailed Metrics Explanation

### Table 1 Metrics

**CPU/GPU Queue Length**: Mean queued jobs per cluster over 24 hours
- Lower is better (less waiting)
- H-MPC target: <350 jobs

**kWh/Job**: Energy efficiency = Total energy (compute + cooling) / Completed jobs
- Lower is better (more efficient)
- Typical range: 2.0-2.5 kWh/job

**Cost ($)**: Operational cost = Energy × electricity price (time-varying)
- Lower is better
- Includes time-of-use pricing (peak/off-peak)

**Max Temp (°C)**: Peak datacenter temperature over episode
- Target: <28°C (safe operating zone)
- Alert: >32°C (throttling onset)

### Additional Metrics (in full results CSV)

- **Utilization (%)**: Mean CPU/GPU usage (target 60-70%)
- **Throttle Time (%)**: Percentage of time above soft threshold
- **Completed Jobs**: Total jobs finished (higher is better)
- **Cost/Job ($)**: Operational cost per completed job

---

## Citation

If you use DataCenterGym or reproduce these results, please cite:

```bibtex
@inproceedings{datacentergym2025,
  title={DataCenterGym: A Physics-Grounded Simulator for Multi-Objective Data Center Scheduling},
  author={Pathak, Nilavra and Biswas, Samadrita and Roy, Nirmalya},
  booktitle={IEEE International Conference on Smart Computing (SMARTCOMP)},
  year={2025}
}
```

---

## Support

For replication issues:
1. Check this file first
2. Review `QUICKSTART.md` for basic setup
3. See `EXPERIMENTS.md` for detailed experiment descriptions
4. Open GitHub issue with:
   - Python version: `python3 --version`
   - Installed packages: `pip freeze`
   - Error message
   - Config file used

**Contact**: npathak@expediagroup.com
