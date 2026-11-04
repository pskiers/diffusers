"""TEMPORARY check (cluster only, no model): do any test mazes also occur in the training data?

Training file: maze_dataset_train_original.parquet (native resolution, with metadata). Test files: <shape>/n<size>_test.parquet.
Compared per test maze: (a) md5 of the maze image bytes (m_original_img) — identical rendering = identical maze;
(b) the generator (shape, width, height, seed); (c) the solution path (start, end, path cell ids) together with shape/size.
Reads the training file in row batches (column subset) to keep memory low.
Usage: python interp_overlap.py <data_root> <test files...>
"""
import hashlib
import json
import sys

import pyarrow.parquet as pq


def img_bytes(v):
    if isinstance(v, dict):
        return v.get("bytes") or b""
    return v if isinstance(v, (bytes, bytearray)) else str(v).encode()


def meta_keys(m):
    d = json.loads(m) if isinstance(m, str) else (m or {})
    mc = d.get("maze_config", {})
    gen = (mc.get("shape"), mc.get("width"), mc.get("height"), mc.get("seed"))
    path = (mc.get("shape"), mc.get("width"), mc.get("height"), str(d.get("start_cell")), str(d.get("end_cell")), tuple(d.get("path_cell_ids", [])))
    return gen, path


def main(root, tests):
    train = pq.ParquetFile(f"{root}/maze_dataset_train_original.parquet")
    print("train rows:", train.metadata.num_rows, "columns:", train.schema_arrow.names, flush=True)
    th, tg, tp, seeds = set(), set(), set(), []
    for batch in train.iter_batches(batch_size=512, columns=["m_original_img", "metadata"]):
        for img, m in zip(batch.column("m_original_img").to_pylist(), batch.column("metadata").to_pylist()):
            th.add(hashlib.md5(img_bytes(img)).hexdigest())
            g, p = meta_keys(m)
            tg.add(g); tp.add(p); seeds.append(g[3])
    s = [x for x in seeds if x is not None]
    print(f"train: {len(th)} distinct maze images, seeds {min(s)}..{max(s)}", flush=True)
    for f in tests:
        t = pq.read_table(f, columns=["m_original_img", "metadata"])
        hs, gs, ps, tseeds = 0, 0, 0, []
        for img, m in zip(t.column("m_original_img").to_pylist(), t.column("metadata").to_pylist()):
            g, p = meta_keys(m)
            hs += hashlib.md5(img_bytes(img)).hexdigest() in th
            gs += g in tg; ps += p in tp; tseeds.append(g[3])
        print(f"{f.split('/maze/')[-1]}: {t.num_rows} test mazes | same maze image in train: {hs} | same generator seed+config: {gs} | "
              f"same solution path: {ps} | test seeds {min(tseeds)}..{max(tseeds)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
