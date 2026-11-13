"""Skip test files whose dependencies are missing on this machine instead of failing at import.
mplib (the motion planner) only exists on Linux; ManiSkill may be absent in a torch-only environment."""
import importlib.util

_has = {m: importlib.util.find_spec(m) is not None for m in ("mani_skill", "mplib")}

collect_ignore = []
if not (_has["mani_skill"] and _has["mplib"]):
    collect_ignore += ["test_planner_gen.py", "test_sim_integration.py", "test_extract.py"]  # all import the planner
