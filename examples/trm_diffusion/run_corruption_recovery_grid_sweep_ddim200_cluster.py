#!/usr/bin/env python3
"""Same corruption-recovery grid as run_corruption_recovery_grid_sweep_ddim100.py,
at num_inference_steps=200 (ddim200) -- meant for a cluster node with more
GPUs and more memory per GPU than the local 8x24GB box this was developed
on. Two things to set before running:

  NUM_GPUS    -- how many GPUs this job actually has. The work-queue
                 dispatcher pattern (shared queue.Queue, one worker thread
                 per GPU pulling jobs via CUDA_VISIBLE_DEVICES) is the same
                 as the local runs; it just needs the real GPU count.
  BATCH_SIZE  -- the local runs were capped at 16 to avoid OOM on 24GB
                 cards; on a 96GB H100 this can go much higher. NOTE: an
                 earlier local test found batch=64 was actually SLOWER than
                 batch=16 on a 24GB card for this attention-heavy backbone
                 (compute-bound on attention FLOPs, not launch-overhead-
                 bound) -- that finding doesn't necessarily transfer to a
                 different GPU class. Worth a quick solo-job timing check
                 at a couple of batch sizes before committing the whole
                 sweep to one value (see the module docstring in
                 corruption_recovery_grid_probe.py for the job structure;
                 a single job like
                   python experiments/corruption_recovery_grid_probe.py \\
                     experiment=mnist_unet_concat_painter_srm_1000t \\
                     +checkpoint=runs/sudoku_unet_concat_t1000-30k.pt \\
                     eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt \\
                     eval.batch_size=<N> sampling.num_inference_steps=200 \\
                     +probe.num_samples=128 +probe.t_starts=[100] \\
                     +probe.n_cells_values=[0,1,2,4,8,16,32,64] \\
                     +probe.conditions=[violating]
                 timed at a couple of eval.batch_size values is enough to
                 pick one).

Job granularity: split by (dataset, t_start, condition, n_cells-half) = 2 x
5 x 2 x 2 = 40 jobs, sorted descending by estimated cost (steps) for
longest-processing-time-first scheduling across NUM_GPUS workers -- the
same load-balancing approach used for the local ddim100 run, just one notch
finer since ddim200's per-t_start step counts (21/61/101/141/181) span an
even wider range and the total run is long enough that a poorly-packed
tail matters more. This does mean the clean (n_cells=0) reference
trajectory gets recomputed once per (condition, n_cells-half) instead of
once per (dataset, t_start) -- roughly 4x redundant instead of ddim100's
2x -- a real but still small cost (clean is a small fraction of each job's
work) traded for much better GPU utilization across the whole run.
"""
import os
import queue
import subprocess
import threading

NUM_GPUS = 8  # <-- SET THIS to the actual GPU count on the cluster node
BATCH_SIZE = 64  # <-- SET THIS after a quick timing check (see docstring)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

COMMON = [
    "experiment=mnist_unet_concat_painter_srm_1000t",
    "+checkpoint=runs/sudoku_unet_concat_t1000-30k.pt",
    "eval_callbacks.0.classifier_path=runs/mnist_classifier_cell16.pt",
    f"eval.batch_size={BATCH_SIZE}",
    "sampling.num_inference_steps=200",
    "+probe.num_samples=128",
]

T_STARTS = [100, 300, 500, 700, 900]  # steps = t_start/5 + 1 at ddim200
N_CELLS_HALVES = {
    "lo": [0, 1, 2, 4],
    "hi": [8, 16, 32, 64],
}

OUT_DIR = "runs/full_ablation/corruption_recovery_grid_ddim200"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("logs/corruption_recovery_grid_ddim200", exist_ok=True)

jobs = []  # (est_steps, label, args)
for dataset_name, dataset_override in [("hard", []), ("easy", ["data=mnist_sudoku"])]:
    base = COMMON + dataset_override
    for t_start in T_STARTS:
        steps = t_start // 5 + 1
        for condition in ["violating", "non_violating"]:
            for half_name, n_cells_values in N_CELLS_HALVES.items():
                label = f"t1000_{dataset_name}_t{t_start}_{condition}_{half_name}"
                args = base + [
                    f"+probe.t_starts=[{t_start}]",
                    f"+probe.conditions=[{condition}]",
                    f"+probe.n_cells_values={n_cells_values}",
                    f"+probe.out={OUT_DIR}/{label}.json",
                ]
                jobs.append((steps, label, args))

jobs.sort(key=lambda j: j[0], reverse=True)  # LPT: longest jobs first into the shared queue
print(f"{len(jobs)} jobs queued across {NUM_GPUS} GPUs, sorted by descending estimated cost")
for steps, label, _ in jobs:
    print(f"  steps={steps:>3}  {label}")

q = queue.Queue()
for j in jobs:
    q.put(j)


def worker(gpu_id):
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    while True:
        try:
            steps, label, args = q.get_nowait()
        except queue.Empty:
            return
        print(f"[gpu {gpu_id}] launching {label} (est_steps={steps})", flush=True)
        with open(f"logs/corruption_recovery_grid_ddim200/{label}.log", "w") as f:
            subprocess.run(
                ["python", "experiments/corruption_recovery_grid_probe.py", *args],
                env=env, stdout=f, stderr=subprocess.STDOUT,
            )
        print(f"[gpu {gpu_id}] finished {label}", flush=True)


threads = [threading.Thread(target=worker, args=(g,)) for g in range(NUM_GPUS)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("ALL CORRUPTION RECOVERY GRID DDIM200 SWEEP JOBS DONE")
