# Vendored GeoSteiner 5.3

Exact Euclidean Steiner minimal tree solver by David M. Warme, Pawel Winter,
and Martin Zachariasen ([geosteiner.com](http://www.geosteiner.com/)),
licensed under **Creative Commons Attribution-NonCommercial 4.0
International** (see `LICENSE`) — academic/non-commercial use only.

Vendored (source only, no prebuilt binaries) from the source bundled in
[kariander1/visual-geo-solver](https://github.com/kariander1/visual-geo-solver)
(the code release for "Visual Diffusion Models are Geometric Solvers",
arXiv 2510.21697), which is what `datasets/steiner_generation.py` adapts to
generate this project's Steiner Tree dataset — see
[[trm-diffusion-new-datasets-roadmap]] / batch 2.

Used only by the **offline, one-time dataset-generation step**
(`datasets/steiner_generation.py`) — solving each instance to optimality
requires the `efst`/`bb` binaries built here. Training/eval never invokes
GeoSteiner directly; they only read the resulting static NDJSON dataset (see
`datasets/steiner_dataset.py`), so a training machine (e.g. a remote cluster)
never needs this directory or a C toolchain at all.

## Build

```
cd third_party/geosteiner-5.3
./build_without_libtool.sh
```

This produces `rand_points`, `efst`, `bb`, `fst2graph` in this directory.
The script provides its own minimal libtool shim and does not require GNU
libtool to be installed — GeoSteiner's Makefile only uses libtool to wrap
building one static library, which the shim reproduces directly with `ar`.
If a real `libtool` is available, plain `make libgeosteiner.la rand_points
efst bb fst2graph` (after `make -C lp_solve_2.3 libLPS.a`) works too.
