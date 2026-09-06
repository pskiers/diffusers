"""Repair `attrdict` 2.0.1 in place for Python >= 3.10 (offline fallback).

Janus imports attrdict (janus/models/modeling_vlm.py, projector.py). The PyPI
release 2.0.1 does `from collections import Mapping`, which was removed in
Python 3.10, so `import janus.models` fails on the 3.11.5 the Helios modules
provide. attrdict3 is the compatible fork, BUT both distributions install the
same top-level `attrdict` package, so if both are present whichever was
installed last owns the files on disk.

Preferred fix (needs PyPI access from the node):

    pip uninstall -y attrdict attrdict3 && pip install attrdict3

Use this script only when the compute node has no outbound network. It rewrites
just the abstract-base-class imports to collections.abc and leaves genuine
`collections` imports (attrdict/dictionary.py's OrderedDict) untouched.
It is idempotent, so re-running is harmless.

    python scripts/fix_attrdict.py            # active venv (sys.prefix)
    python scripts/fix_attrdict.py "$VENV"    # explicit prefix

NB the Helios venv is aarch64, so run this on a gh200 compute node via srun,
not on the login node.
"""
import glob
import os
import re
import sys

ABC_NAMES = {"Mapping", "MutableMapping", "Sequence", "MutableSequence",
             "Iterable", "Iterator", "Callable", "Set", "MutableSet"}


def _rewrite(match: "re.Match") -> str:
    names = [n.strip() for n in match.group(1).split(",") if n.strip()]
    abc = [n for n in names if n in ABC_NAMES]
    rest = [n for n in names if n not in ABC_NAMES]
    lines = []
    if rest:
        lines.append("from collections import " + ", ".join(rest))
    if abc:
        lines.append("from collections.abc import " + ", ".join(abc))
    return "\n".join(lines)


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else sys.prefix
    roots = glob.glob(os.path.join(prefix, "lib*", "python3.*",
                                   "site-packages", "attrdict"))
    if not roots:
        print(f"no attrdict package found under {prefix}", file=sys.stderr)
        return 1

    changed = 0
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "*.py"))):
            src = open(path).read()
            out = re.sub(r"^from collections import ([A-Za-z_][A-Za-z_0-9, ]*)$",
                         _rewrite, src, flags=re.M)
            if out != src:
                open(path, "w").write(out)
                print("patched", path)
                changed += 1

    print(f"done ({changed} file(s) changed)")
    if changed == 0:
        print("nothing to do - attrdict already imports cleanly, "
              "or attrdict3 is the copy on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
