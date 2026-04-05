"""
SQL Query Optimization Environment
An OpenEnv-compliant environment where an AI agent learns to optimize SQL queries.
"""

import sqlite3
import json
import os
from typing import Optional
from dataclasses import dataclass, field, asdict

# --- Try pydantic (needed for FastAPI); fall back to stdlib dataclasses ------

try:
    from pydantic import BaseModel, Field as PField

    class SQLAction(BaseModel):
        query: str = PField(..., description="SQL query to submit.")
        message: str = PField(default="", description="Explanation of optimization.")

    class SQLObservation(BaseModel):
        echoed_message: str
        task_description: str
        original_query: str
        schema_info: str
        last_query_result: Optional[str] = None
        last_query_error: Optional[str] = None
        step: int = 0
        done: bool = False
        score: float = 0.0

    class SQLReward(BaseModel):
        value: float
        reason: str

    _PYDANTIC = True

except ImportError:
    _PYDANTIC = False

    @dataclass
    class SQLAction:
        query: str
        message: str = ""

    @dataclass
    class SQLObservation:
        echoed_message: str
        task_description: str
        original_query: str
        schema_info: str
        last_query_result: Optional[str] = None
        last_query_error: Optional[str] = None
        step: int = 0
        done: bool = False
        score: float = 0.0

        def model_dump(self):
            return asdict(self)

    @dataclass
    class SQLReward:
        value: float
        reason: str


# --- Schema & Seed Data --------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL,
    manager_id INTEGER REFERENCES employees(id)
);
CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    budget REAL NOT NULL,
    location TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT,
    budget REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS project_assignments (
    employee_id INTEGER REFERENCES employees(id),
    project_id INTEGER REFERENCES projects(id),
    role TEXT NOT NULL,
    hours_per_week REAL NOT NULL,
    PRIMARY KEY (employee_id, project_id)
);
"""

SEED_SQL = """
INSERT OR IGNORE INTO departments VALUES (1,'Engineering',500000,'New York');
INSERT OR IGNORE INTO departments VALUES (2,'Marketing',200000,'San Francisco');
INSERT OR IGNORE INTO departments VALUES (3,'Sales',300000,'Chicago');
INSERT OR IGNORE INTO departments VALUES (4,'HR',150000,'New York');
INSERT OR IGNORE INTO employees VALUES (1,'Alice Johnson','Engineering',120000,'2019-03-15',NULL);
INSERT OR IGNORE INTO employees VALUES (2,'Bob Smith','Engineering',95000,'2020-07-01',1);
INSERT OR IGNORE INTO employees VALUES (3,'Carol White','Marketing',85000,'2018-11-20',NULL);
INSERT OR IGNORE INTO employees VALUES (4,'Dave Brown','Sales',75000,'2021-02-10',NULL);
INSERT OR IGNORE INTO employees VALUES (5,'Eve Davis','Engineering',110000,'2020-01-05',1);
INSERT OR IGNORE INTO employees VALUES (6,'Frank Miller','HR',70000,'2017-06-30',NULL);
INSERT OR IGNORE INTO employees VALUES (7,'Grace Wilson','Marketing',80000,'2019-09-14',3);
INSERT OR IGNORE INTO employees VALUES (8,'Hank Moore','Sales',72000,'2022-03-01',4);
INSERT OR IGNORE INTO employees VALUES (9,'Iris Taylor','Engineering',130000,'2016-08-22',1);
INSERT OR IGNORE INTO employees VALUES (10,'Jack Anderson','Sales',68000,'2023-01-15',4);
INSERT OR IGNORE INTO projects VALUES (1,'Cloud Migration','Engineering','2023-01-01','2024-06-30',250000);
INSERT OR IGNORE INTO projects VALUES (2,'Brand Refresh','Marketing','2023-03-01','2023-12-31',80000);
INSERT OR IGNORE INTO projects VALUES (3,'CRM Upgrade','Sales','2023-06-01',NULL,120000);
INSERT OR IGNORE INTO projects VALUES (4,'AI Chatbot','Engineering','2024-01-01',NULL,180000);
INSERT OR IGNORE INTO project_assignments VALUES (1,1,'Lead',40);
INSERT OR IGNORE INTO project_assignments VALUES (2,1,'Developer',30);
INSERT OR IGNORE INTO project_assignments VALUES (5,1,'Developer',25);
INSERT OR IGNORE INTO project_assignments VALUES (9,4,'Lead',40);
INSERT OR IGNORE INTO project_assignments VALUES (2,4,'Developer',20);
INSERT OR IGNORE INTO project_assignments VALUES (3,2,'Lead',35);
INSERT OR IGNORE INTO project_assignments VALUES (7,2,'Analyst',20);
INSERT OR IGNORE INTO project_assignments VALUES (4,3,'Lead',30);
INSERT OR IGNORE INTO project_assignments VALUES (8,3,'Sales Rep',25);
"""

TASKS = [
    {
        "id": "task_easy",
        "name": "Fix SELECT * and Add WHERE clause",
        "difficulty": "easy",
        "description": (
            "The query below fetches ALL employees using SELECT *. "
            "Optimize it by: (1) selecting only id, name, department, salary columns, "
            "(2) adding a WHERE clause to filter only Engineering department employees, "
            "(3) adding ORDER BY salary DESC."
        ),
        "original_query": "SELECT * FROM employees;",
        "expected_columns": {"id", "name", "department", "salary"},
        "required_clauses": ["where", "order by"],
        "required_filters": ["engineering"],
        "min_rows": 1,
        "max_rows": 50,
    },
    {
        "id": "task_medium",
        "name": "Optimize N+1 with JOIN",
        "difficulty": "medium",
        "description": (
            "The original query retrieves employees then does a subquery per row to get their project count --- "
            "a classic N+1 problem. Rewrite it as a single query using LEFT JOIN + GROUP BY to get each "
            "employee's name, department, salary, and the number of projects they are assigned to (as project_count). "
            "Include employees with 0 projects. Order by project_count DESC."
        ),
        "original_query": (
            "SELECT name, department, salary, "
            "(SELECT COUNT(*) FROM project_assignments pa WHERE pa.employee_id = e.id) AS project_count "
            "FROM employees e;"
        ),
        "expected_columns": {"name", "department", "salary", "project_count"},
        "required_clauses": ["join", "group by", "order by"],
        "required_filters": [],
        "min_rows": 5,
        "max_rows": 50,
    },
    {
        "id": "task_hard",
        "name": "Complex Aggregation with Subquery",
        "difficulty": "hard",
        "description": (
            "The original query is broken --- it tries to find employees earning above their department average "
            "but uses AVG() in a WHERE clause which is invalid SQL. "
            "Fix and optimize it: return employee name, department, salary, and the department average salary "
            "(as dept_avg_salary). Only include employees whose salary is strictly above their department average. "
            "Order by department, then salary DESC."
        ),
        "original_query": (
            "SELECT name, department, salary, AVG(salary) "
            "FROM employees "
            "WHERE salary > AVG(salary);"
        ),
        "expected_columns": {"name", "department", "salary", "dept_avg_salary"},
        "required_clauses": ["group by", "order by"],
        "required_filters": [],
        "min_rows": 1,
        "max_rows": 50,
    },
]


# --- Environment ---------------------------------------------------------------

class SQLOptimizationEnv:
    """OpenEnv-compliant environment for SQL query optimization tasks."""

    MAX_STEPS = 10

    def __init__(self, task_id: Optional[str] = None):
        self._task_id = task_id
        self._task: Optional[dict] = None
        self._step_count = 0
        self._done = False
        self._score = 0.0
        self._conn: Optional[sqlite3.Connection] = None
        self._history: list = []
        self._last_result: Optional[str] = None
        self._last_error: Optional[str] = None

    def _init_db(self):
        if self._conn:
            self._conn.close()
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._conn.executescript(SEED_SQL)
        self._conn.commit()

    def _execute_query(self, query: str):
        try:
            cur = self._conn.execute(query)
            rows = cur.fetchall()
            return [dict(r) for r in rows], None
        except Exception as e:
            return None, str(e)

    def _get_schema_info(self) -> str:
        rows, _ = self._execute_query(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        if not rows:
            return "No schema available."
        return "\n\n".join(f"-- Table: {r['name']}\n{r['sql']}" for r in rows)

    def _grade(self, query: str, task: dict):
        q_lower = query.lower().strip()
        rows, error = self._execute_query(query)
        if error:
            return 0.0, f"Execution failed: {error}"
        if rows is None:
            return 0.0, "No result."

        score = 0.0
        reasons = []

        # Required clauses (0.30)
        required = task.get("required_clauses", [])
        if required:
            hit = sum(1 for c in required if c in q_lower)
            cs = (hit / len(required)) * 0.30
        else:
            cs = 0.30
        score += cs
        reasons.append(f"clauses:{cs:.2f}/0.30")

        # Required filters (0.15)
        filters = task.get("required_filters", [])
        if filters:
            hit = sum(1 for f in filters if f.lower() in q_lower)
            fs = (hit / len(filters)) * 0.15
        else:
            fs = 0.15
        score += fs
        reasons.append(f"filters:{fs:.2f}/0.15")

        # No SELECT * (0.10)
        if "select *" not in q_lower:
            score += 0.10
            reasons.append("no_select_*:+0.10")

        # Expected columns (0.25)
        if rows:
            returned_cols = set(rows[0].keys())
            expected = task.get("expected_columns", set())
            if expected:
                matched = len(expected & returned_cols)
                col_s = (matched / len(expected)) * 0.25
                score += col_s
                reasons.append(f"cols:{matched}/{len(expected)}:{col_s:.2f}/0.25")
            else:
                score += 0.25

        # Row count (0.20)
        min_r, max_r = task.get("min_rows", 0), task.get("max_rows", 9999)
        if min_r <= len(rows) <= max_r:
            score += 0.20
            reasons.append(f"rows:{len(rows)}in[{min_r},{max_r}]:+0.20")
        else:
            reasons.append(f"rows:{len(rows)}out_of_range:+0.00")

        return round(min(score, 1.0), 4), " | ".join(reasons)

    def reset(self, task_id: Optional[str] = None) -> SQLObservation:
        self._init_db()
        self._step_count = 0
        self._done = False
        self._score = 0.0
        self._history = []
        self._last_result = None
        self._last_error = None

        tid = task_id or self._task_id or "task_easy"
        task_map = {t["id"]: t for t in TASKS}
        self._task = task_map.get(tid, TASKS[0])

        return SQLObservation(
            echoed_message=f"Environment reset. Task: {self._task['name']}",
            task_description=self._task["description"],
            original_query=self._task["original_query"],
            schema_info=self._get_schema_info(),
            step=0,
            done=False,
            score=0.0,
        )

    def step(self, action: SQLAction):
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")

        self._step_count += 1
        query = action.query.strip()

        rows, error = self._execute_query(query)
        if error:
            self._last_error = error
            self._last_result = None
        else:
            self._last_error = None
            self._last_result = json.dumps(rows[:5], indent=2) if rows else "[]"

        score, grade_reason = self._grade(query, self._task)
        self._score = score

        prev_score = max((h.get("score", 0.0) for h in self._history), default=0.0)
        delta = score - prev_score
        reward = round(max(-1.0, min(1.0, delta - 0.02)), 4)

        self._history.append({"step": self._step_count, "query": query, "score": score, "reward": reward})

        done = score >= 0.95 or self._step_count >= self.MAX_STEPS
        self._done = done

        msg = f"Step {self._step_count}: score={score:.3f}, reward={reward:+.3f}. {grade_reason}"
        if error:
            msg += f" | ERROR: {error}"

        obs = SQLObservation(
            echoed_message=msg,
            task_description=self._task["description"],
            original_query=self._task["original_query"],
            schema_info=self._get_schema_info(),
            last_query_result=self._last_result,
            last_query_error=self._last_error,
            step=self._step_count,
            done=done,
            score=score,
        )
        return obs, reward, done, {"grade_reason": grade_reason, "score": score}

    def state(self) -> dict:
        return {
            "task_id": self._task["id"] if self._task else None,
            "task_name": self._task["name"] if self._task else None,
            "step": self._step_count,
            "done": self._done,
            "score": self._score,
            "history_length": len(self._history),
        }
