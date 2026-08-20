#!/bin/bash
# Builds the 4 GeoSteiner binaries needed by datasets/steiner_generation.py
# (rand_points, efst, bb, fst2graph) without depending on GNU libtool being
# installed. GeoSteiner's Makefile normally shells out to libtool for a
# handful of --mode=compile/--mode=link invocations that only exist to wrap
# building a static library — this script provides a minimal drop-in shim for
# exactly those two invocation shapes and nothing else, then drives the same
# Makefile with LIBTOOL overridden to point at it. Plain `make` (with a real
# libtool installed) works too, if you have one available.
#
# Usage: cd third_party/geosteiner-5.3 && ./build_without_libtool.sh
set -euo pipefail
cd "$(dirname "$0")"

# Force the autoconf-generated files' mtimes into dependency order so Make
# never decides configure/config.status/Makefile need regenerating (this
# sandbox has no autoconf/automake to do that with, and a fresh git checkout
# normally gives every file the same mtime, which can go either way). This is
# a no-op correctness-wise: config.h/Makefile are already fully generated and
# checked in, we just need Make to see them as up to date.
touch -d '2020-01-01 00:00:00' configure.ac aclocal.m4 Makefile.in config.h.in \
  functions.in parmdefs.h errordefs.h geosteiner_config.in
touch -d '2020-01-01 00:00:05' configure
touch -d '2020-01-01 00:00:10' config.status
touch -d '2020-01-01 00:00:15' config.h stamp-config-h Makefile geosteiner.h

SHIM_DIR="$(mktemp -d)"
cat > "$SHIM_DIR/libtool" <<'EOF'
#!/bin/bash
set -e
mode=""
args=()
for a in "$@"; do
  case "$a" in
    --mode=*) mode="${a#--mode=}" ;;
    --tag=*) ;;
    *) args+=("$a") ;;
  esac
done

if [ "$mode" = "compile" ]; then
  out=""
  newargs=()
  prev=""
  for a in "${args[@]}"; do
    if [ "$prev" = "-o" ]; then
      out="$a"
      newargs+=("${out%.lo}.o")
    else
      newargs+=("$a")
    fi
    prev="$a"
  done
  "${newargs[@]}"
  touch "$out"
  exit 0
fi

if [ "$mode" = "link" ]; then
  objs=()
  outla=""
  found_o=false
  for a in "${args[@]}"; do
    if $found_o; then
      outla="$a"
      found_o=false
      continue
    fi
    case "$a" in
      -o) found_o=true ;;
      *.lo) objs+=("${a%.lo}.o") ;;
      *) ;;
    esac
  done
  mkdir -p .libs
  ar rcs ".libs/${outla%.la}.a" "${objs[@]}"
  touch "$outla"
  exit 0
fi

echo "libtool shim: unhandled mode '$mode' args: $*" >&2
exit 1
EOF
chmod +x "$SHIM_DIR/libtool"

make lp_solve_2.3/libLPS.a
make LIBTOOL="$SHIM_DIR/libtool --tag=CC" libgeosteiner.la
make LIBTOOL="$SHIM_DIR/libtool --tag=CC" rand_points efst bb fst2graph

rm -rf "$SHIM_DIR"
echo "Built: rand_points efst bb fst2graph"
