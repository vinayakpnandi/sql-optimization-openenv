"""
Tests for SQLOptimizationEnv — verifies all 3 tasks have working graders
and scores in the 0.0–1.0 range.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from env.environment import SQLOptimizationEnv, SQLAction


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return SQLOptimizationEnv()


# ── Task: easy ────────────────────────────────────────────────────────────────

def test_easy_reset(env):
    obs = env.reset(task_id="task_easy")
    assert obs.task_description
    assert obs.original_query == "SELECT * FROM employees;"
    assert obs.step == 0
    assert not obs.done


def test_easy_bad_query_scores_low(env):
    env.reset(task_id="task_easy")
    obs, reward, done, info = env.step(SQLAction(query="SELECT * FROM employees;"))
    assert 0.0 <= info["score"] <= 1.0
    assert info["score"] < 0.6, "SELECT * with no WHERE should score below 0.6"


def test_easy_good_query_scores_high(env):
    env.reset(task_id="task_easy")
    obs, reward, done, info = env.step(SQLAction(
        query="SELECT id, name, department, salary FROM employees WHERE department = 'Engineering' ORDER BY salary DESC;",
        message="Selected specific columns, filtered by department, added ORDER BY"
    ))
    assert 0.0 <= info["score"] <= 1.0
    assert info["score"] >= 0.7, f"Optimized query should score >= 0.7, got {info['score']}"


# ── Task: medium ──────────────────────────────────────────────────────────────

def test_medium_reset(env):
    obs = env.reset(task_id="task_medium")
    assert "N+1" in obs.task_description or "JOIN" in obs.task_description or "project" in obs.task_description.lower()
    assert obs.step == 0


def test_medium_bad_query_scores_low(env):
    env.reset(task_id="task_medium")
    obs, reward, done, info = env.step(SQLAction(
        query="SELECT name, department, salary, (SELECT COUNT(*) FROM project_assignments pa WHERE pa.employee_id = e.id) AS project_count FROM employees e;"
    ))
    assert 0.0 <= info["score"] <= 1.0
    # Correlated subquery still returns correct columns, so some partial credit OK
    assert info["score"] < 0.8, "Unoptimized N+1 query should score below 0.8"


def test_medium_good_query_scores_high(env):
    env.reset(task_id="task_medium")
    obs, reward, done, info = env.step(SQLAction(
        query="""
        SELECT e.name, e.department, e.salary, COUNT(pa.project_id) AS project_count
        FROM employees e
        LEFT JOIN project_assignments pa ON e.id = pa.employee_id
        GROUP BY e.id, e.name, e.department, e.salary
        ORDER BY project_count DESC;
        """,
        message="Replaced correlated subquery with LEFT JOIN + GROUP BY"
    ))
    assert 0.0 <= info["score"] <= 1.0
    assert info["score"] >= 0.75, f"Optimized JOIN query should score >= 0.75, got {info['score']}"


# ── Task: hard ────────────────────────────────────────────────────────────────

def test_hard_reset(env):
    obs = env.reset(task_id="task_hard")
    assert obs.task_description
    assert obs.step == 0


def test_hard_broken_query_fails(env):
    env.reset(task_id="task_hard")
    obs, reward, done, info = env.step(SQLAction(
        query="SELECT name, department, salary, AVG(salary) FROM employees WHERE salary > AVG(salary);"
    ))
    assert 0.0 <= info["score"] <= 1.0
    assert info["score"] < 0.5, "Broken aggregation query should score below 0.5"


def test_hard_good_query_scores_high(env):
    env.reset(task_id="task_hard")
    obs, reward, done, info = env.step(SQLAction(
        query="""
        SELECT e.name, e.department, e.salary, dept_avg.avg_salary AS dept_avg_salary
        FROM employees e
        JOIN (
            SELECT department, AVG(salary) AS avg_salary
            FROM employees
            GROUP BY department
        ) dept_avg ON e.department = dept_avg.department
        WHERE e.salary > dept_avg.avg_salary
        ORDER BY e.department, e.salary DESC;
        """,
        message="Used subquery to compute dept avg, filtered employees above avg, added ORDER BY"
    ))
    assert 0.0 <= info["score"] <= 1.0
    assert info["score"] >= 0.65, f"Correct aggregation query should score >= 0.65, got {info['score']}"


# ── General ───────────────────────────────────────────────────────────────────

def test_reward_in_range(env):
    env.reset(task_id="task_easy")
    _, reward, _, _ = env.step(SQLAction(query="SELECT id, name FROM employees WHERE department='HR' ORDER BY name;"))
    assert -1.0 <= reward <= 1.0


def test_done_on_max_steps(env):
    env.reset(task_id="task_easy")
    done = False
    for _ in range(10):
        _, _, done, _ = env.step(SQLAction(query="SELECT 1;"))
        if done:
            break
    assert done, "Episode should be done after MAX_STEPS"


def test_state_returns_dict(env):
    env.reset(task_id="task_medium")
    state = env.state()
    assert isinstance(state, dict)
    assert "task_id" in state
    assert "score" in state


def test_score_range_all_tasks(env):
    """All graders must produce scores in 0.0–1.0."""
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        env.reset(task_id=task_id)
        _, _, _, info = env.step(SQLAction(query="SELECT 1;"))
        assert 0.0 <= info["score"] <= 1.0, f"{task_id} score out of range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
