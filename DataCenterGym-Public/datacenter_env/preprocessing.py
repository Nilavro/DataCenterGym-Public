import pandas as pd
import numpy as np
from collections import defaultdict

def load_machine_meta(path):
    try:
        df = pd.read_csv(path, header=None)
        df.columns = ["machine_id", "timestamp", "rack_id", "field4", "cpu_percent", "mem_percent", "status"]
    except Exception as e:
        print("⚠️ Error loading machine_meta.csv, generating minimal machine metadata.")
        usage_df = pd.read_csv("alibaba_data/machine_usage.csv", header=None)
        usage_df.columns = [
            "machine_id", "timestamp", "cpu_util_percent", "mem_util_percent",
            "disk_io", "net_io", "cpu_percent", "mem_percent", "num_running_tasks"
        ]
        df = pd.DataFrame({"machine_id": usage_df["machine_id"].drop_duplicates()})
    return df

def group_machines_to_clusters(meta_df, num_clusters=5):
    unique_machines = meta_df['machine_id'].unique()
    cluster_map = {m: i % num_clusters for i, m in enumerate(unique_machines)}
    return cluster_map

def load_batch_tasks(path, nrows=None):
    """
    Load batch task traces.

    Args:
        path: Path to batch_task.csv
        nrows: Optional limit on number of rows (for testing). Use None for all rows.
    """
    print(f"   Loading CSV (this may take a minute)...")
    df = pd.read_csv(path, header=None, nrows=nrows)
    print(f"   Loaded {len(df)} rows")

    df.columns = [
        "task_name", "task_id", "job_id", "instance_num",
        "status", "start_time", "end_time", "cpu_request", "priority"
    ]
    df = df.dropna(subset=["start_time", "task_id", "instance_num"])
    df["start_time"] = df["start_time"].astype(int)

    print(f"   After filtering: {len(df)} rows")
    return df

def construct_job_trace(tasks_df, max_time, target_load=0.75, time_step_minutes=5, use_contiguous_window=True, duration_scale=4.5, max_jobs_per_step=None):
    """
    Construct job arrival trace from task dataframe with improved methodology.

    Args:
        tasks_df: DataFrame with task information
        max_time: Maximum time step to consider (e.g., 288 for 24 hours)
        target_load: Target system load (0.0-1.0), controls Bernoulli thinning probability
        time_step_minutes: Duration of each timestep in minutes (default: 5)
        use_contiguous_window: If True, select contiguous time window; if False, use old scaling approach
        duration_scale: Factor to scale down job durations (default: 4.5 to match original bundle characteristics)
        max_jobs_per_step: Optional cap on jobs per timestep (applied after thinning, preserves burstiness up to cap)
    """
    print(f"   Constructing job trace (max_time={max_time}, target_load={target_load})...")

    TIME_STEP_SECONDS = time_step_minutes * 60

    tasks_df = tasks_df.copy()

    # Select time window
    if len(tasks_df) > 0:
        if use_contiguous_window:
            # Use contiguous window to preserve burstiness and temporal correlations
            # Define window in RAW TRACE SECONDS, then map to simulator timesteps
            min_start = tasks_df["start_time"].min()
            max_start = tasks_df["start_time"].max()
            trace_duration = max_start - min_start

            # Define 24-hour window in raw trace seconds
            WINDOW_DURATION_SECONDS = 24 * 3600  # 24 hours in seconds

            if trace_duration > WINDOW_DURATION_SECONDS:
                # Select contiguous 24-hour window starting at 25% into trace
                window_start = min_start + int(0.25 * trace_duration)
                window_end = window_start + WINDOW_DURATION_SECONDS

                # Filter tasks to this window
                tasks_df = tasks_df[(tasks_df["start_time"] >= window_start) &
                                   (tasks_df["start_time"] < window_end)].copy()

                # Map to simulator timesteps: floor((start_time - window_start) / 300)
                tasks_df["start_time"] = ((tasks_df["start_time"] - window_start) / TIME_STEP_SECONDS).astype(int)

                # Keep only tasks that fit in episode
                tasks_df = tasks_df[tasks_df["start_time"] < max_time].copy()

                print(f"   Selected 24h window [{window_start}, {window_end}) in raw seconds")
                print(f"   Mapped to simulator timesteps [0, {max_time})")
            else:
                # Trace shorter than 24h, use entire trace and map to simulator time
                tasks_df["start_time"] = ((tasks_df["start_time"] - min_start) / TIME_STEP_SECONDS).astype(int)
                tasks_df = tasks_df[tasks_df["start_time"] < max_time].copy()
                print(f"   Using entire trace ({trace_duration} seconds), mapped to simulator time")
        else:
            # Old approach: time compression (kept for backward compatibility)
            min_start = tasks_df["start_time"].min()
            max_start = tasks_df["start_time"].max()
            time_range = max_start - min_start
            if time_range > 0:
                tasks_df["start_time"] = ((tasks_df["start_time"] - min_start) / time_range * max_time).astype(int)
                tasks_df = tasks_df[tasks_df["start_time"] < max_time]
            print(f"   WARNING: Using time compression (destroys burstiness)")

    print(f"   Tasks in time window: {len(tasks_df)}")

    # Vectorized job type assignment
    # NOTE: Alibaba 2018 trace predates GPU workloads - we synthetically assign GPU affinity
    # to create heterogeneous resource demands representative of modern ML-intensive datacenters
    np.random.seed(42)  # For reproducibility
    tasks_df["resource_demand"] = tasks_df["cpu_request"].fillna(1.0)

    # 40% CPU, 60% GPU split (synthetic for Alibaba trace)
    random_vals = np.random.rand(len(tasks_df))
    tasks_df["tau"] = np.where(random_vals < 0.4, "cpu", "gpu")

    # Bernoulli thinning for load control (replaces arbitrary max_jobs_per_step capping)
    # This preserves burstiness while controlling overall arrival rate
    np.random.seed(43)  # Different seed for thinning
    keep_mask = np.random.rand(len(tasks_df)) < target_load
    tasks_df = tasks_df[keep_mask].copy()
    print(f"   After Bernoulli thinning (p={target_load}): {len(tasks_df)} tasks retained")

    # Build job trace
    job_trace = defaultdict(list)
    job_id = 0

    # Group by time for efficiency
    grouped = tasks_df.groupby("start_time")
    total_times = len(grouped)

    for i, (t, group) in enumerate(grouped):
        if i % 100 == 0:  # Progress every 100 time steps
            print(f"   Progress: {i}/{total_times} time steps processed")

        for _, row in group.iterrows():
            # Unit-based duration conversion (replaces magic number /3000)
            # Original Alibaba durations are in seconds, convert to timesteps
            end_time = row.get("end_time", None)

            if end_time is None or pd.isna(end_time):
                # No end_time available, use minimum duration
                scaled_duration = 1
            else:
                raw_duration_seconds = int(end_time - row["start_time"])

                # Convert to simulation timesteps with scaling
                # duration_steps = ceil((seconds / timestep_seconds) / duration_scale)
                # Scaling is needed because Alibaba jobs run for days, but our episodes are 24h
                duration_in_timesteps = raw_duration_seconds / TIME_STEP_SECONDS
                scaled_duration = max(1, int(np.ceil(duration_in_timesteps / duration_scale)))

                # Cap at episode length for practical reasons
                scaled_duration = min(scaled_duration, max_time)

            job = {
                "id": int(job_id),
                "r": float(row["resource_demand"]),
                "d": scaled_duration,
                "v": float(row.get("priority", 1)),
                "tau": row["tau"],
            }
            job_trace[t].append(job)
            job_id += 1

    print(f"   Job trace complete: {job_id} jobs across {len(job_trace)} time steps")

    # Apply max_jobs_per_step cap if specified (preserves burstiness up to the cap)
    if max_jobs_per_step is not None:
        total_jobs_before = sum(len(jobs) for jobs in job_trace.values())
        np.random.seed(44)  # Different seed for per-step capping
        for t in job_trace.keys():
            if len(job_trace[t]) > max_jobs_per_step:
                # Randomly sample max_jobs_per_step jobs from this timestep
                job_trace[t] = list(np.random.choice(job_trace[t], size=max_jobs_per_step, replace=False))
        total_jobs_after = sum(len(jobs) for jobs in job_trace.values())
        jobs_dropped = total_jobs_before - total_jobs_after
        print(f"   Applied max_jobs_per_step={max_jobs_per_step} cap: {total_jobs_after} jobs retained ({jobs_dropped} dropped)")

    # Report statistics
    all_durations = [job["d"] for jobs in job_trace.values() for job in jobs]
    if all_durations:
        print(f"   Duration statistics: min={min(all_durations)}, mean={np.mean(all_durations):.1f}, max={max(all_durations)}")

    arrivals_per_step = [len(jobs) for jobs in job_trace.values()]
    if arrivals_per_step:
        print(f"   Arrivals per step: min={min(arrivals_per_step)}, mean={np.mean(arrivals_per_step):.1f}, max={max(arrivals_per_step)}")

    return job_trace

def load_usage_trace(path):
    df = pd.read_csv(path, header=None)
    df.columns = [
        "machine_id", "timestamp", "cpu_util_percent", "mem_util_percent",
        "disk_io", "net_io", "cpu_percent", "mem_percent", "num_running_tasks"
    ]
    df["timestamp"] = df["timestamp"].astype(int)
    return df[["timestamp", "machine_id", "cpu_util_percent"]]

def aggregate_usage_by_cluster(usage_df, cluster_map, num_clusters):
    usage_df["cluster_id"] = usage_df["machine_id"].map(cluster_map)
    grouped = usage_df.groupby(["timestamp", "cluster_id"])["cpu_util_percent"].mean().reset_index()
    usage_trace = defaultdict(lambda: [0.0] * num_clusters)
    for _, row in grouped.iterrows():
        t = int(row["timestamp"])
        c = int(row["cluster_id"])
        usage_trace[t][c] = float(row["cpu_util_percent"])
    return usage_trace
