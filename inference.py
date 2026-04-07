"""
Baseline Inference Script — SQL Optimization OpenEnv
=====================================================
Reads API credentials from environment variables:
  API_BASE_URL  - LLM endpoint (OpenAI-compatible)
  API_KEY       - API key
  MODEL_NAME    - model identifier
  HF_TOKEN      - Hugging Face token (for Space access)

Produces structured [START] / [STEP] / [END] stdout logs as required.
"""

import asyncio
import json
import os
import sys
import time
from typing import Optional

from openai import AsyncOpenAI

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY: str = os.environ.get("API_KEY", "sk-placeholder")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
IMAGE_NAME: str = os.environ.get("IMAGE_NAME", "sql-optimization-openenv:latest")

TASK_NAME: str = "sql_optimization"
BENCHMARK: str = "sql-openenv"
MAX_STEPS: int = 10
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 512

# ── Logging helpers (required format) ────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(json.dumps({"type": "START", "task": task, "env": env, "model": model}), flush=True)

def log_step(step: int, action: dict, observation: dict, reward: float, done: bool):
    print(json.dumps({
        "type": "STEP",
        "step": step,
        "action": action,
        "observation": observation,
        "reward": reward,
        "done": done,
    }), flush=True)

def log_end(task: str, score: float, steps: int, success: bool):
    print(json.dumps({
        "type": "END",
        "task": task,
        "score": score,
        "steps": steps,
        "success": success,
    }), flush=True)

# ── Local env shim (avoids needing a running Docker container for baseline) ───

class MyEnvV4Env:
    """Thin wrapper around the local environment for baseline testing."""
    def __init__(self):
        sys.path.insert(0, os.path.dirname(__file__))
        from env.environment import SQLOptimizationEnv, SQLAction as _SQLAction
        self._env = SQLOptimizationEnv()
        self._SQLAction = _SQLAction

    async def reset(self):
        obs = self._env.reset()
        return _ObsWrapper(obs)

    async def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return _StepResult(obs, reward, done, info)


class _ObsWrapper:
    def __init__(self, obs):
        self.observation = obs


class _StepResult:
    def __init__(self, obs, reward, done, info):
        self.observation = obs
        self.reward = reward
        self.done = done
        self.info = info


class MyEnvV4Action:
    """Compatibility shim matching competition interface."""
    def __init__(self, message: str):
        sys.path.insert(0, os.path.dirname(__file__))
        from env.environment import SQLAction
        # Parse the message as JSON or plain SQL
        try:
            data = json.loads(message)
            self._action = SQLAction(**data)
        except Exception:
            self._action = SQLAction(query=message, message="")

    def __new__(cls, message: str):
        obj = object.__new__(cls)
        obj.__init__(message)
        return obj._action


# ── LLM call ─────────────────────────────────────────────────────────────────

async def get_model_message(
    client: AsyncOpenAI,
    step: int,
    last_echoed: str,
    last_reward: float,
    history: list[str],
) -> str:
    system_prompt = (
        "You are an expert SQL optimization agent. "
        "You will be given a poorly written SQL query and a database schema. "
        "Your job is to rewrite the query to be correct, efficient, and follow best practices. "
        "Respond ONLY with a JSON object: {\"query\": \"<optimized SQL>\", \"message\": \"<brief explanation>\"}. "
        "No markdown, no explanation outside the JSON."
    )
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    for h in history:
        messages.append({"role": "user", "content": h})
    messages.append({"role": "user", "content": f"Step {step}. Last observation: {last_echoed}\nLast reward: {last_reward}"})

    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else '{"query": "SELECT 1", "message": "fallback"}'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"query": "SELECT 1", "message": "error fallback"}'


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Run all 3 tasks
    task_ids = ["task_easy", "task_medium", "task_hard"]
    all_scores = []

    for task_id in task_ids:
        env = MyEnvV4Env()

        history: list[str] = []
        rewards: list[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=f"{TASK_NAME}/{task_id}", env=BENCHMARK, model=MODEL_NAME)

        try:
            result = await env.reset()
            obs = result.observation
            # Include task context in first history entry
            history.append(
                f"Task: {obs.task_description}\n\nOriginal Query:\n{obs.original_query}\n\nSchema:\n{obs.schema_info}"
            )
            last_echoed = obs.echoed_message
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                if hasattr(obs, "done") and obs.done:
                    break

                message = await get_model_message(client, step, last_echoed, last_reward, history)

                from env.environment import SQLAction
                try:
                    action_data = json.loads(message)
                    action = SQLAction(**action_data)
                except Exception:
                    action = SQLAction(query=message, message="")

                result = await env.step(action)
                obs = result.observation
                reward = result.reward
                done = result.done

                steps_taken = step
                rewards.append(reward)
                score = obs.score
                last_echoed = obs.echoed_message
                last_reward = reward

                history.append(f"Observation: {obs.echoed_message}")
                if obs.last_query_result:
                    history.append(f"Query result (first 5 rows): {obs.last_query_result}")
                if obs.last_query_error:
                    history.append(f"Query error: {obs.last_query_error}")

                log_step(
                    step=step,
                    action=action.model_dump(),
                    observation=obs.model_dump(),
                    reward=reward,
                    done=done,
                )

                if done:
                    break

            success = score >= 0.90
            all_scores.append(score)

        except Exception as exc:
            print(f"[DEBUG] Episode error: {exc}", flush=True)
            all_scores.append(0.0)

        log_end(task=f"{TASK_NAME}/{task_id}", score=score, steps=steps_taken, success=success)

    print(json.dumps({"type": "SUMMARY", "scores": dict(zip(task_ids, all_scores)), "mean": round(sum(all_scores)/len(all_scores), 4)}), flush=True)


def entry_point():
    """Entry point for package installation that handles async main."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
