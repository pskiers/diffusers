# Vendored GLEW header

`GL/glew.h` from [nigels-com/glew](https://github.com/nigels-com/glew), release
`glew-2.1.0` (BSD-style license, see `LICENSE.txt`).

Only the header is vendored, not the library — `free-mujoco-py==2.1.6`
already bundles a prebuilt `libglewegl.so` (see its
`mujoco_py/binaries/linux/mujoco210/bin/`), but its sdist is missing the
`GL/glew.h` header that `mujoco_py/gl/eglshim.c` needs at compile time (a
packaging bug: this header is generated from GLEW's own build process and
only ships in GLEW's release tarballs, not its git tree, so it's easy to
miss when repackaging). `eglshim.c` only uses the plain `glewInit()`/
`glewGetErrorString()` API, which is stable across GLEW versions, so any
reasonably close release header works against the bundled `.so`.

Used by `scripts/patch_mujoco_py_gpu.py`, which copies this file into the
installed `mujoco_py` package's `vendor/egl/GL/` include directory (already
on its build's include path) rather than requiring network access or a
system-wide GLEW install at patch time.
