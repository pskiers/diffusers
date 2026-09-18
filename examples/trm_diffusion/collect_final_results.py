"""Collect AMAZE metric JSONs into final_results/<task>/<model>/metrics.json.

The scorer writes to different places depending on the path:
  * DiT / PT   -> runs/<RUN_NAME>/amaze_metrics_<task>.json          (one file, all shapes)
  * VLM queens -> <GEN_DIR>/amaze_metrics_queens.json                (one file)
  * VLM maze   -> <GEN_DIR>/amaze_metrics_maze_<shape>.json          (ONE FILE PER SHAPE)

so maze VLM results have to be merged across the four shape files.
"""
import json
import os
import shutil
import sys
import time

ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
OUT = os.path.join(ROOT, "final_results")
SHAPES = ["square", "hexagon", "triangle", "circle"]
EAR = "third_party/ear-amaze/inference_results"
BAGEL_Q = "runs/bagel_infer/bagel_queens_2026.09.16_10.19.55_50/generated_images"

# (task, model, kind, location)
#   kind "single" -> location is a path to one json
#   kind "shards" -> location is a GEN_DIR holding amaze_metrics_maze_<shape>.json
# NB: the DiT/PT script ignores RUN_NAME and writes into the CHECKPOINT's run dir,
# so those paths are the checkpoint dirs, not runs/final_*.
SOURCES = [
    ("maze",   "dit",        "single", "runs/dit_maze_final/amaze_metrics_maze.json"),
    ("maze",   "pt",         "single", "runs/pt_maze_final_thinker/amaze_metrics_maze.json"),
    ("maze",   "janus",      "shards", EAR + "/maze"),
    ("maze",   "bagel",      "shards", "runs/final_bagel_maze"),
    ("queens", "dit",        "single", "runs/queens_dit_baseline/amaze_metrics_queens.json"),
    ("queens", "pt",         "single", "runs/pt_queens_final_thinker/amaze_metrics_queens.json"),
    ("queens", "janus",      "single", EAR + "/queens_std/amaze_metrics_queens.json"),
    ("queens", "janus_test", "single", EAR + "/queens_n4/amaze_metrics_queens.json"),
    ("queens", "bagel",      "single", BAGEL_Q + "/amaze_metrics_queens.json"),
]
# NOTE: maze/janus_test is deliberately absent -- those generations were overwritten
# on the cluster and are unrecoverable (confirmed 2026-09-17).

# Shards from one batch can legitimately finish hours apart (separate queue slots),
# so this is a prompt to check, not a refusal.
SHARD_SPREAD_WARN_H = 24

KEYS = ("violation", "coverage", "mse_inside", "mse_outside",
        "pass1", "pass5", "exact1", "exact5")


OOD_KEYS = ("overall_ood", "per_shape_ood", "per_geometry_ood", "per_scale_ood")

# Maze scoring ignores a neighbourhood of this many cell widths around the given
# start/goal (eval/amaze_eval.py::MAZE_ENDPOINT_TOLERANCE). It changes the numbers
# WITHOUT changing the schema, so the exact1 key-presence check below cannot see it --
# every maze json now carries the tolerance it was scored with, and one that does not
# match is refused rather than shipped alongside results that do.
EXPECTED_ENDPOINT_TOLERANCE = float(os.environ.get("MAZE_ENDPOINT_TOLERANCE", "1.3"))


def endpoint_mismatch(data):
    """None if this maze result was scored at the expected tolerance, else a reason."""
    if data.get("task") != "maze":
        return None
    got = data.get("endpoint_tolerance")
    if got is None:
        return ("scored before the endpoint-tolerance patch (no 'endpoint_tolerance' "
                "key); its maze numbers use the old asymmetric start/goal credit")
    if abs(float(got) - EXPECTED_ENDPOINT_TOLERANCE) > 1e-9:
        return "scored at endpoint_tolerance=%s, expected %s" % (got, EXPECTED_ENDPOINT_TOLERANCE)
    return None


def strip_ood(data):
    """Drop OOD sections unconditionally.

    MAZE_OOD_SCALES=/QUEEN_OOD_SCALES= only take effect if sbatch propagates the
    (empty) env var to the job. Rather than depend on that, guarantee the shipped
    metrics carry no OOD section either way.
    """
    removed = [k for k in OOD_KEYS if k in data]
    for k in removed:
        del data[k]
    return removed


def is_stale(data):
    """True if ANY size row predates the Exact patch.

    Must scan every row, not just the first: a merged maze result can be partly
    refreshed (one shape re-scored, three not), and sampling a single row would
    let that through with exact1 silently defaulting to 0.0.
    """
    rows = []
    for shp in data.get("per_shape", {}).values():
        rows.extend(shp.values())
    rows.extend(data.get("per_scale", {}).values())
    if not rows:
        rows = [data.get("overall", {})]
    return any("exact1" not in r for r in rows)


def stale_detail(data):
    """Which (geometry, size) rows are still pre-patch -- for a useful message."""
    bad = []
    for geom, shp in data.get("per_shape", {}).items():
        for size, row in shp.items():
            if "exact1" not in row:
                bad.append("%s/%s" % (geom, size))
    for size, row in data.get("per_scale", {}).items():
        if "exact1" not in row:
            bad.append("n%s" % size)
    return bad


def mean_rows(rows):
    if not rows:
        return dict((k, 0.0) for k in KEYS)
    return dict((k, sum(r.get(k, 0.0) for r in rows) / float(len(rows))) for k in KEYS)


def merge_shards(gen_dir):
    """Combine per-shape maze jsons into one result with the canonical schema.

    Also records each shard's mtime. Shards live in GEN_DIR and are overwritten
    in place, so re-running a subset silently mixes old and new results. The
    exact1 check catches that only because this patch added a key; a future
    change that alters numbers without changing the schema would be invisible.
    Timestamps catch it regardless of schema.
    """
    per_shape, per_geometry, found = {}, {}, []
    stamps = {}
    tolerances = {}
    for shape in SHAPES:
        for cand in (os.path.join(gen_dir, shape, "amaze_metrics_maze_%s.json" % shape),
                     os.path.join(gen_dir, "amaze_metrics_maze_%s.json" % shape)):
            if os.path.isfile(cand):
                with open(cand) as fh:
                    d = json.load(fh)
                per_shape.update(d.get("per_shape", {}))
                per_geometry.update(d.get("per_geometry", {}))
                stamps[shape] = os.path.getmtime(cand)
                tolerances[shape] = d.get("endpoint_tolerance")
                found.append(shape)
                break
    if not found:
        return None, [], {}
    rows = [sz for shp in per_shape.values() for sz in shp.values()]
    # A shard that disagrees is reported as None so endpoint_mismatch() refuses the
    # whole merge -- shards are scored by separate jobs and can straddle a patch.
    distinct = set(tolerances.get(s) for s in found)
    return {
        "task": "maze",
        "endpoint_tolerance": distinct.pop() if len(distinct) == 1 else None,
        "overall": mean_rows(rows),
        "per_shape": per_shape,
        "per_geometry": per_geometry,
        "merged_from_shapes": found,
        "shard_endpoint_tolerances": tolerances,
        "shard_mtimes": dict((k, time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(v)))
                             for k, v in stamps.items()),
    }, found, stamps


def drop_stale_output(dst_dir, task, model):
    """Remove a previously collected result we are refusing to refresh this run.

    Collecting only ever WROTE, so a skipped source silently left the last run's
    metrics.json in place and it was zipped anyway -- the shipped bundle could mix
    metrics from different scorers. final_results/ is a derived directory, fully
    rebuilt from the run dirs, so dropping the stale copy is safe and makes the zip
    honest: what is absent was not re-scored.
    """
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
        print("        removed stale final_results/%s/%s (not refreshed this run)"
              % (task, model))


def main():
    made = []
    for task, model, kind, loc in SOURCES:
        dst_dir = os.path.join(OUT, task, model)
        src = os.path.join(ROOT, loc)
        if kind == "single":
            if not os.path.isfile(src):
                print("MISSING %-7s %-11s %s" % (task, model, loc))
                drop_stale_output(dst_dir, task, model)
                continue
            with open(src) as fh:
                data = json.load(fh)
            note = loc
            if is_stale(data):
                print("STALE   %-7s %-11s %s -- pre-patch json (no exact1); SKIPPED"
                      % (task, model, loc))
                drop_stale_output(dst_dir, task, model)
                continue
            why = endpoint_mismatch(data)
            if why:
                print("STALE   %-7s %-11s %s -- %s; SKIPPED" % (task, model, loc, why))
                drop_stale_output(dst_dir, task, model)
                continue
        else:
            data, found, stamps = merge_shards(src)
            if data is None:
                print("MISSING %-7s %-11s %s (no shape jsons)" % (task, model, loc))
                drop_stale_output(dst_dir, task, model)
                continue
            note = "%s (shapes: %s)" % (loc, ", ".join(found))
            if is_stale(data):
                bad = stale_detail(data)
                print("STALE   %-7s %-11s %s -- %d pre-patch rows; SKIPPED"
                      % (task, model, loc, len(bad)))
                print("        still pre-patch: %s" % ", ".join(sorted(bad)[:8]))
                drop_stale_output(dst_dir, task, model)
                continue
            why = endpoint_mismatch(data)
            if why:
                print("STALE   %-7s %-11s %s -- %s; SKIPPED" % (task, model, loc, why))
                print("        shard tolerances: %s" % data.get("shard_endpoint_tolerances"))
                drop_stale_output(dst_dir, task, model)
                continue
            if len(found) < len(SHAPES):
                print("PARTIAL %-7s %-11s only %d/%d shapes: %s"
                      % (task, model, len(found), len(SHAPES), ", ".join(found)))
            if stamps:
                spread = max(stamps.values()) - min(stamps.values())
                stamp_txt = "  ".join(
                    "%s=%s" % (k, time.strftime("%m-%d %H:%M", time.localtime(v)))
                    for k, v in sorted(stamps.items(), key=lambda kv: kv[1]))
                print("        shard times: %s  (spread %.1fh)" % (stamp_txt, spread / 3600.0))
                if spread > SHARD_SPREAD_WARN_H * 3600:
                    print("        !! shards span >%dh -- verify this was ONE batch, not a "
                          "partial re-run" % SHARD_SPREAD_WARN_H)

        removed = strip_ood(data)
        if removed:
            print("        (stripped OOD sections: %s)" % ", ".join(removed))
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        with open(os.path.join(dst_dir, "metrics.json"), "w") as fh:
            json.dump(data, fh, indent=2)
        with open(os.path.join(dst_dir, "source.txt"), "w") as fh:
            fh.write("source: %s\n" % note)
            if kind == "single":
                fh.write("mtime: %s\n" % time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(src))))
            for shp, st in sorted(data.get("shard_mtimes", {}).items()):
                fh.write("shard %-9s %s\n" % (shp, st))
        print("OK      %-7s %-11s <- %s" % (task, model, note))
        made.append(os.path.join(task, model))

    print("")
    print("collected %d of %d" % (len(made), len(SOURCES)))
    if len(made) < len(SOURCES):
        print("(MISSING = job not finished yet; STALE = pre-patch json, re-run that job)")
    for m in made:
        print("  final_results/%s/metrics.json" % m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
