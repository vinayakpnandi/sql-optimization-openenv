"""
Pre-Submission Validation Script
Checks all items from the Pre-Submission Checklist locally
before pushing to HF Spaces.
"""
import subprocess
import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

PASS = "✅"
FAIL = "❌"
results = []

def check(name, fn):
    try:
        ok, detail = fn()
        results.append((name, ok, detail))
        symbol = PASS if ok else FAIL
        print(f"{symbol} {name}: {detail}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"{FAIL} {name}: EXCEPTION — {e}")


# ── 1. openenv.yaml exists and has required fields ───────────────────────────
def check_yaml():
    import yaml
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    required = ["name", "observation_space", "action_space", "tasks", "endpoints"]
    missing = [k for k in required if k not in data]
    if missing:
        return False, f"Missing fields: {missing}"
    return True, f"Valid — {len(data['tasks'])} tasks defined"

check("openenv.yaml valid", check_yaml)


# ── 2. Dockerfile exists ─────────────────────────────────────────────────────
def check_dockerfile():
    ok = os.path.exists("Dockerfile")
    return ok, "Dockerfile found" if ok else "Dockerfile missing"

check("Dockerfile exists", check_dockerfile)


# ── 3. inference.py exists at root ───────────────────────────────────────────
def check_inference():
    ok = os.path.exists("inference.py")
    return ok, "inference.py found at root" if ok else "inference.py missing from root"

check("inference.py at root", check_inference)


# ── 4. Environment can reset ─────────────────────────────────────────────────
def check_reset():
    from env.environment import SQLOptimizationEnv
    env = SQLOptimizationEnv()
    obs = env.reset()
    ok = obs.echoed_message is not None and obs.original_query
    return ok, f"reset() returned observation with echoed_message='{obs.echoed_message[:50]}...'"

check("reset() works", check_reset)


# ── 5. All 3 tasks produce scores in 0.0–1.0 ─────────────────────────────────
def check_graders():
    from env.environment import SQLOptimizationEnv, SQLAction
    env = SQLOptimizationEnv()
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        env.reset(task_id=task_id)
        _, _, _, info = env.step(SQLAction(query="SELECT 1;"))
        s = info["score"]
        if not (0.0 <= s <= 1.0):
            return False, f"Task {task_id} score={s} out of range"
    return True, "All 3 tasks produce scores in [0.0, 1.0]"

check("3+ tasks with graders (0.0–1.0)", check_graders)


# ── 6. step() and state() return typed Pydantic models ───────────────────────
def check_typed_models():
    from env.environment import SQLOptimizationEnv, SQLAction, SQLObservation
    env = SQLOptimizationEnv()
    env.reset()
    obs, reward, done, info = env.step(SQLAction(query="SELECT id FROM employees LIMIT 1;"))
    assert isinstance(obs, SQLObservation), "step() must return SQLObservation"
    assert isinstance(reward, float), "reward must be float"
    assert isinstance(done, bool), "done must be bool"
    state = env.state()
    assert isinstance(state, dict), "state() must return dict"
    return True, "step() → SQLObservation, reward: float, done: bool; state() → dict"

check("Typed models (Pydantic)", check_typed_models)


# ── 7. Pytest suite passes ────────────────────────────────────────────────────
def check_tests():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        capture_output=True, text=True
    )
    passed = result.returncode == 0
    last_line = result.stdout.strip().split("\n")[-1] if result.stdout else result.stderr[:200]
    return passed, last_line

check("pytest suite passes", check_tests)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"Pre-Submission Validation: {passed}/{total} checks passed")
if passed == total:
    print("🎉 All checks passed — ready to submit!")
else:
    print("⚠️  Fix the failing checks above before submitting.")
    sys.exit(1)
