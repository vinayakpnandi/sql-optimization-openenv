# SQL Optimization OpenEnv 🗄️⚡

An **OpenEnv-compliant** reinforcement learning environment where an AI agent learns to optimize SQL queries. The agent receives poorly-written SQL (with `SELECT *`, N+1 subqueries, broken aggregations) and must rewrite them to be correct, performant, and follow best practices.

---

## 🎯 Motivation

SQL query optimization is a genuine, high-value real-world skill. Poor queries cause slow dashboards, expensive cloud bills, and data bugs. This environment tests whether an agent can:
1. Understand a database schema
2. Identify anti-patterns in SQL (SELECT *, correlated subqueries, missing GROUP BY)
3. Rewrite queries iteratively using feedback rewards

---

## 🏗️ Environment Description

| Property | Value |
|---|---|
| **Action Space** | `SQLAction(query: str, message: str)` |
| **Observation Space** | `SQLObservation` (see below) |
| **Reward Range** | `[-1.0, 1.0]` |
| **Max Steps / Episode** | 10 |
| **Tasks** | 3 (easy → medium → hard) |

### Observation Space

```python
class SQLObservation(BaseModel):
    echoed_message: str        # Status message / grading feedback
    task_description: str      # What the agent needs to do
    original_query: str        # The broken/unoptimized query to fix
    schema_info: str           # Full database schema (CREATE TABLE statements)
    last_query_result: str     # JSON of first 5 rows from last query (nullable)
    last_query_error: str      # Error if last query failed (nullable)
    step: int                  # Current step number
    done: bool                 # Episode ended?
    score: float               # Current score [0.0–1.0]
```

### Action Space

```python
class SQLAction(BaseModel):
    query: str      # The optimized SQL query to submit
    message: str    # Brief explanation of what was changed (optional)
```

### Reward Function

The reward at each step is:
```
reward = (current_score - previous_best_score) - 0.02 * step_penalty
```

The score is computed by a multi-criteria grader:
- **Clause usage** (0.30): Uses required SQL clauses (WHERE, GROUP BY, JOIN, ORDER BY, etc.)
- **Filter correctness** (0.15): Required filter values present
- **No SELECT *** (0.10): Avoids selecting all columns
- **Expected columns** (0.25): Returns the required output columns
- **Row count sanity** (0.20): Result set is in expected range

---

## 📋 Tasks

### Task 1 — Easy: Fix SELECT * and Add WHERE Clause
- **Original**: `SELECT * FROM employees;`
- **Goal**: Select only `id, name, department, salary`; filter `WHERE department = 'Engineering'`; add `ORDER BY salary DESC`
- **Expected Score**: ≥ 0.70 for correct solution

### Task 2 — Medium: Eliminate N+1 Subquery with JOIN
- **Original**: Correlated subquery inside SELECT per employee
- **Goal**: Rewrite using `LEFT JOIN + GROUP BY` to count projects per employee in one query
- **Expected Score**: ≥ 0.75 for correct solution

### Task 3 — Hard: Fix Broken Aggregation
- **Original**: `SELECT ... AVG(salary) FROM employees WHERE salary > AVG(salary);` ← broken SQL
- **Goal**: Use a subquery or CTE to compute department averages, then filter employees above their dept avg
- **Expected Score**: ≥ 0.65 for correct solution

---

## 🚀 Setup & Usage

### Prerequisites

```bash
python 3.11+
pip install -r requirements.txt
```

### Run Locally

```bash
# Start the API server
python app.py
# Server runs on http://localhost:7860
```

### Run with Docker

```bash
docker build -t sql-openenv .
docker run -p 7860:7860 sql-openenv
```

### API Endpoints

```
POST /reset    — Start a new episode
POST /step     — Submit an action (SQL query)
GET  /state    — Get current environment state
GET  /tasks    — List available tasks
GET  /health   — Health check (returns 200)
```

#### Example: Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'
```

#### Example: Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<session_id_from_reset>",
    "action": {
      "query": "SELECT id, name, department, salary FROM employees WHERE department = '\''Engineering'\'' ORDER BY salary DESC;",
      "message": "Fixed SELECT *, added WHERE and ORDER BY"
    }
  }'
```

---

## 🤖 Running the Baseline Inference Script

Set environment variables then run:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="hf_..."

python inference.py
```

The script will run all 3 tasks sequentially and print structured `[START]`, `[STEP]`, and `[END]` JSON logs.

### Baseline Scores (gpt-4o-mini)

| Task | Difficulty | Score |
|------|-----------|-------|
| task_easy | Easy | ~0.85 |
| task_medium | Medium | ~0.75 |
| task_hard | Hard | ~0.65 |

---

## ✅ Pre-Submission Validation

```bash
pip install pytest pyyaml
python validate.py
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
sql-openenv/
├── app.py              # FastAPI application (OpenEnv HTTP API)
├── inference.py        # Baseline inference script (required by competition)
├── openenv.yaml        # OpenEnv metadata spec
├── Dockerfile          # Container definition for HF Spaces
├── requirements.txt    # Python dependencies
├── validate.py         # Pre-submission validation script
├── README.md           # This file
├── env/
│   ├── __init__.py
│   └── environment.py  # Core environment implementation
└── tests/
    └── test_environment.py  # Pytest tests for all 3 tasks
```

---

## 🏷️ Environment Tags

`openenv` `sql` `optimization` `real-world` `code` `database`
