"""Compatibility entrypoint for pre-schedule DSE hook.

The backend pipeline invokes:
    python -m dse.forward_bound

The implementation lives in `dse/src/forward_bound.py`.
This file forwards execution so existing compiled backends keep working.
"""

from dse.src.forward_bound import _load_yaml_with_fallback, main, run_forward_bound


if __name__ == "__main__":
    main()

