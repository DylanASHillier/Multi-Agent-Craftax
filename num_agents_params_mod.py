# num_agents_params_mod.py
import pickle, re
from typing import Any, Iterable, Tuple, List, Optional, Dict
from collections import Counter
from collections.abc import Mapping, Sequence

# Optional: support NumPy + JAX arrays (either or both)
try:
    import numpy as np
except Exception:
    np = None
try:
    import jax.numpy as jnp  # type: ignore
except Exception:
    jnp = None

Path = Tuple[str, ...]

def _is_array(x: Any) -> bool:
    return ((np is not None and isinstance(x, np.ndarray)) or
            (jnp is not None and isinstance(x, jnp.ndarray)))

def _np_like(x):
    if jnp is not None and isinstance(x, jnp.ndarray):
        return jnp
    return np  # may be None

def _walk(obj: Any, prefix: Path = ()) -> Iterable[Tuple[Path, Any]]:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from _walk(v, prefix + (repr(k),))
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for i, v in enumerate(obj):
            yield from _walk(v, prefix + (f"[{i}]",))
    else:
        yield prefix, obj

def _path_str(path: Path) -> str:
    return ".".join(path).replace(".[", "[")

def _resize_first_dim(x: Any, old_n: int, new_n: int, jitter: float = 0.0):
    """
    Resize array along axis 0 by slicing (shrink) or tiling (grow).
    Preserves NumPy vs JAX type. Optional tiny noise when growing.
    """
    lib = _np_like(x)
    if lib is None or getattr(x, "ndim", 0) < 1 or int(x.shape[0]) != int(old_n):
        return x

    if new_n == old_n:
        y = x
    elif new_n < old_n:
        y = x[:new_n, ...]
    else:
        reps = new_n // old_n
        rem = new_n % old_n
        parts = [x] * reps + ([x[:rem, ...]] if rem else [])
        y = lib.concatenate(parts, axis=0) if len(parts) > 1 else x

    if jitter and new_n > old_n:
        try:
            std = float(lib.std(y)) if hasattr(lib, "std") else 1.0
            noise = (std * 1e-3 * float(jitter)) * lib.random.normal(size=y.shape)  # type: ignore[attr-defined]
            y = y + noise
        except Exception:
            pass
    return y

def _update_int_keys(root: Any, new_n: int, key_regex: str) -> Tuple[Any, List[Tuple[str, int, int]]]:
    """
    Overwrite int leaves whose KEY PATH matches regex with new_n.
    Returns (new_root, changes).
    """
    rx = re.compile(key_regex, re.IGNORECASE)
    changes: List[Tuple[str, int, int]] = []

    def _recurse(obj: Any, prefix: Path = ()):
        if isinstance(obj, Mapping):
            changed = False
            out = {}
            for k, v in obj.items():
                p = prefix + (repr(k),)
                if isinstance(v, int) and rx.search(_path_str(p)):
                    if v != new_n:
                        changes.append((_path_str(p), v, new_n))
                    out[k] = new_n
                    changed = True
                else:
                    nv = _recurse(v, p)
                    changed = changed or (nv is not v)
                    out[k] = nv
            return out if changed else obj

        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            changed = False
            items = []
            for i, v in enumerate(obj):
                nv = _recurse(v, prefix + (f"[{i}]",))
                changed = changed or (nv is not v)
                items.append(nv)
            if not changed:
                return obj
            return tuple(items) if isinstance(obj, tuple) else list(items)

        return obj

    new_root = _recurse(root, ())
    return new_root, changes

class NumAgentsParamsMod:
    """
    Load a pickle, set the number of agents (resize tensors with leading dim == old_n
    and optionally update any config ints like num_agents), then save.
    """

    def __init__(self, root: Any):
        self.root = root

    @classmethod
    def from_pickle(cls, path: str) -> "NumAgentsParamsMod":
        # ⚠️ Only unpickle trusted files.
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return cls(obj)

    def to_pickle(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.root, f, protocol=pickle.HIGHEST_PROTOCOL)

    def guess_old_agents(self, min_dim: int = 2) -> Optional[int]:
        counts = Counter()
        for _, leaf in _walk(self.root):
            if _is_array(leaf) and getattr(leaf, "ndim", 0) >= 1:
                d0 = int(leaf.shape[0])
                if d0 >= min_dim:
                    counts[d0] += 1
        return counts.most_common(1)[0][0] if counts else None

    def set_num_agents(
        self,
        new_n: int,
        old_n: Optional[int] = None,
        *,
        update_config_ints: bool = True,
        key_regex: str = r"(?:^|\.)(num_agents|n_agents|numPlayers|num_players)(?:$|\.)",
        jitter: float = 0.0,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Resize all tensors whose first dim == old_n to new_n and optionally update
        integer config fields matching key_regex to new_n.

        Returns a report dict with details of planned/applied changes.
        """
        if old_n is None:
            old_n = self.guess_old_agents()
        if old_n is None:
            return {"ok": False, "error": "Could not auto-detect agent count; pass old_n."}

        tensor_plan: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
        for path, leaf in _walk(self.root):
            if _is_array(leaf) and getattr(leaf, "ndim", 0) >= 1 and int(leaf.shape[0]) == int(old_n):
                nsh = (new_n,) + tuple(leaf.shape[1:])
                if tuple(leaf.shape) != nsh:
                    tensor_plan.append((_path_str(path), tuple(leaf.shape), nsh))

        int_plan: List[Tuple[str, int, int]] = []
        if update_config_ints:
            # compute plan without mutating: run updater on a temp reference
            _, int_plan = _update_int_keys(self.root, new_n, key_regex)

        if dry_run:
            return {
                "ok": True,
                "old_n": old_n,
                "new_n": new_n,
                "tensor_changes": tensor_plan,
                "int_changes": int_plan,
                "applied": False,
            }

        # Apply tensor changes
        def _apply(obj: Any, prefix: Path = ()):
            if _is_array(obj) and getattr(obj, "ndim", 0) >= 1:
                return _resize_first_dim(obj, old_n, new_n, jitter=jitter)
            if isinstance(obj, Mapping):
                changed = False
                out = {}
                for k, v in obj.items():
                    nv = _apply(v, prefix + (repr(k),))
                    changed = changed or (nv is not v)
                    out[k] = nv
                return out if changed else obj
            if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
                changed = False
                items = []
                for i, v in enumerate(obj):
                    nv = _apply(v, prefix + (f"[{i}]",))
                    changed = changed or (nv is not v)
                    items.append(nv)
                if not changed:
                    return obj
                return tuple(items) if isinstance(obj, tuple) else list(items)
            return obj

        self.root = _apply(self.root, ())

        # Apply int updates
        if update_config_ints:
            self.root, int_plan = _update_int_keys(self.root, new_n, key_regex)

        return {
            "ok": True,
            "old_n": old_n,
            "new_n": new_n,
            "tensor_changes": tensor_plan,
            "int_changes": int_plan,
            "applied": True,
        }
