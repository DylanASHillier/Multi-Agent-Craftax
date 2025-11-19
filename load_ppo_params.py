#!/usr/bin/env python3
# walk_pickle_leaves.py
import pickle, argparse
from typing import Any, Iterable, Tuple
from collections.abc import Mapping, Sequence

try:
    import numpy as np
except Exception:
    np = None
try:
    import jax.numpy as jnp  # type: ignore
except Exception:
    jnp = None

def _is_array(x: Any) -> bool:
    return ((np is not None and isinstance(x, np.ndarray)) or
            (jnp is not None and isinstance(x, jnp.ndarray)))

Path = Tuple[str, ...]

def iter_leaves(obj: Any, prefix: Path = (), _seen: set[int] | None = None
               ) -> Iterable[Tuple[Path, Any]]:
    """Yield (path, leaf_value) for every leaf in dict/list/tuple structures."""
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        yield prefix + ("<shared-ref>",), obj
        return
    _seen.add(oid)

    if _is_array(obj):
        yield prefix, obj
        return

    if isinstance(obj, Mapping):
        if not obj:
            yield prefix, obj
            return
        for k, v in obj.items():
            yield from iter_leaves(v, prefix + (repr(k),), _seen)
        return

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        if len(obj) == 0:
            yield prefix, obj
            return
        for i, v in enumerate(obj):
            yield from iter_leaves(v, prefix + (f"[{i}]",), _seen)
        return

    yield prefix, obj  # scalars, strings, objects, etc.

def _path_str(p: Path) -> str:
    return ".".join(p).replace(".[", "[")

def main():
    ap = argparse.ArgumentParser(description="Walk a pickle and print every leaf path.")
    ap.add_argument("path", help="Path to .p/.pkl file (e.g., params-ippo-v4.p)")
    args = ap.parse_args()

    with open(args.path, "rb") as f:
        obj = pickle.load(f)  # NOTE: only load pickles you trust

    for path, leaf in iter_leaves(obj):
        p = _path_str(path)
        if _is_array(leaf):
            shape = getattr(leaf, "shape", None)
            dtype = getattr(leaf, "dtype", None)
            print(f"{p} = <array shape={shape} dtype={dtype}>")
        else:
            s = repr(leaf)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"{p} = {type(leaf).__name__}: {s}")

if __name__ == "__main__":
    main()
