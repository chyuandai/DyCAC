#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_DIR = PROJECT_ROOT / "framework_codes"
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

try:
    from sotopia.agents import LLMAgent
    from sotopia.messages import AgentAction, Observation
    _SOTOPIA_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    LLMAgent = object
    AgentAction = None
    Observation = Any
    _SOTOPIA_IMPORT_ERROR = exc

app = typer.Typer(pretty_exceptions_enable=False, help="Run the framework through SOTOPIA benchmark.")

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return float(raw)

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return int(raw)

def _compact_text(value: Any) -> str:
    """Convert SOTOPIA observations or profiles to robust plain text."""
    if value is None:
        return ""
    to_nl = getattr(value, "to_natural_language", None)
    if callable(to_nl):
        try:
            return str(to_nl())
        except Exception:
            pass
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return str(model_dump())
        except Exception:
            pass
    return str(value)

def _profile_name(agent_profile: Any) -> str:
    first = str(getattr(agent_profile, "first_name", "") or "").strip()
    last = str(getattr(agent_profile, "last_name", "") or "").strip()
    full = " ".join(part for part in (first, last) if part).strip()
    return full or str(getattr(agent_profile, "pk", "") or "SOTOPIA agent")

def _profile_persona(agent_profile: Any) -> str:
    name = _profile_name(agent_profile)
    parts = [name]
    occupation = str(getattr(agent_profile, "occupation", "") or "").strip()
    public_info = str(getattr(agent_profile, "public_info", "") or "").strip()
    gender_pronoun = str(getattr(agent_profile, "gender_pronoun", "") or "").strip()
    decision_style = str(getattr(agent_profile, "decision_making_style", "") or "").strip()
    if occupation:
        parts.append(f"occupation: {occupation}")
    if gender_pronoun:
        parts.append(f"pronouns: {gender_pronoun}")
    if decision_style:
        parts.append(f"decision style: {decision_style}")
    if public_info:
        parts.append(f"public background: {public_info}")
    return "; ".join(parts)

def _bootstrap_memory(agent_profile: Any) -> dict[str, Any]:
    """Seed the framework memory from the official SOTOPIA AgentProfile."""
    return {
        "world_facts": {
            "self_name": _profile_name(agent_profile),
            "self_age": getattr(agent_profile, "age", ""),
            "self_occupation": getattr(agent_profile, "occupation", ""),
            "self_gender": getattr(agent_profile, "gender", ""),
            "self_pronouns": getattr(agent_profile, "gender_pronoun", ""),
            "self_background": getattr(agent_profile, "public_info", ""),
            "self_big_five": getattr(agent_profile, "big_five", ""),
            "self_moral_values": getattr(agent_profile, "moral_values", []),
            "self_schwartz_values": getattr(agent_profile, "schwartz_personal_values", []),
            "self_personality_and_values": getattr(agent_profile, "personality_and_values", ""),
            "self_decision_style": getattr(agent_profile, "decision_making_style", ""),
            "self_secret": getattr(agent_profile, "secret", ""),
            "self_mbti": getattr(agent_profile, "mbti", ""),
        },
        "mental_state": {
            "desires": {},
            "beliefs": {},
            "intentions": {},
        },
        "dialogue_meta": {
            "turn_count": 0,
            "recent_turns": [],
            "dominant_topic_seq": [],
            "belief_revision_log": [],
            "user_model_of_ai": None,
        },
    }

def _infer_goal_from_observation(observation_text: str) -> str:
    """Best-effort goal extraction; the full observation is still passed to the framework."""
    patterns: Iterable[str] = (
        r"(?:your|you\s+need\s+to\s+achieve\s+the)\s+goal\s*(?:is|:|：)\s*(.+?)(?:\n|$)",
        r"(?:social\s+goal|private\s+goal|goal)\s*(?:is|:|：)\s*(.+?)(?:\n|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, observation_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            goal = re.sub(r"\s+", " ", match.group(1)).strip()
            return goal[:500]
    return ""

def _choose_action_type(observation: Any, response: str) -> str:
    actions = list(getattr(observation, "available_actions", []) or [])
    if not response.strip() and "none" in actions:
        return "none"
    if "speak" in actions:
        return "speak"
    if "action" in actions:
        return "action"
    if "none" in actions:
        return "none"
    return "speak"

class FrameworkSotopiaAgent(LLMAgent):
    """SOTOPIA-compatible custom agent backed by this repository's framework."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._agent_profile = kwargs.get("agent_profile")
        self._benchmark_model_name = kwargs.get("model_name", "")
        if _SOTOPIA_IMPORT_ERROR is None:
            super().__init__(*args, **kwargs)
            self._agent_profile = getattr(self, "agent_profile", self._agent_profile)
            self._benchmark_model_name = getattr(self, "model_name", self._benchmark_model_name)

        self.memory: dict[str, Any] = _bootstrap_memory(self._agent_profile)
        self.cultural_state: dict[str, Any] | None = None
        self.timestep = 0
        self.base_url = os.getenv("FRAMEWORK_BASE_URL", "")
        self.api_key = os.getenv("FRAMEWORK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.framework_model_name = os.getenv("FRAMEWORK_MODEL_NAME") or str(self._benchmark_model_name or "gpt-4o")
        self.seed = _env_int("FRAMEWORK_SEED", 42)
        self.debug = _env_bool("FRAMEWORK_DEBUG", False)
        self.temperature_perception = _env_float("FRAMEWORK_TEMP_PERCEPTION", 0.2)
        self.temperature_memory = _env_float("FRAMEWORK_TEMP_MEMORY", 0.1)
        self.temperature_culture = _env_float("FRAMEWORK_TEMP_CULTURE", 0.3)
        self.temperature_response = _env_float("FRAMEWORK_TEMP_RESPONSE", 0.7)
        self.n_hypotheses = _env_int("FRAMEWORK_N_HYPOTHESES", 5)
        self.agent_persona = _profile_persona(self._agent_profile)
        self._last_goal = os.getenv("FRAMEWORK_SOCIAL_GOAL", "")

    async def aact(self, observation: Observation) -> Any:
        if AgentAction is None:
            raise RuntimeError(f"SOTOPIA is not importable: {_SOTOPIA_IMPORT_ERROR}")
        observation_text = _compact_text(observation)
        response = await asyncio.to_thread(self._generate_response, observation_text)
        action_type = _choose_action_type(observation, response)
        if action_type == "none":
            return AgentAction(action_type="none", argument="")
        return AgentAction(action_type=action_type, argument=response.strip())

    def _generate_response(self, observation_text: str) -> str:
        from main import ingest_input, run_pipeline

        social_goal = self._last_goal or _infer_goal_from_observation(observation_text)
        if social_goal:
            self._last_goal = social_goal

        current_input = (
            f"[Official SOTOPIA observation]\n{observation_text}\n\n"
            "Task: respond as the current SOTOPIA character. Follow the available action constraints "
            "from the observation. Use the framework's cultural adaptation pipeline, but do not reveal "
            "private chain-of-thought or hidden framework state."
        )
        current_input, current_memory = ingest_input(current_input, self.memory, timestep=self.timestep)
        response, self.memory, self.cultural_state, _ = run_pipeline(
            current_input=current_input,
            prior_memory=current_memory,
            prior_cultural_state=self.cultural_state,
            timestep=self.timestep,
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.framework_model_name,
            temperature_perception=self.temperature_perception,
            temperature_memory=self.temperature_memory,
            temperature_culture=self.temperature_culture,
            temperature_response=self.temperature_response,
            seed=self.seed,
            n_hypotheses=self.n_hypotheses,
            debug=self.debug,
            social_goal=social_goal,
            agent_persona=self.agent_persona,
        )
        self.timestep += 1
        return response or ""

def _ensure_sotopia_available() -> None:
    if _SOTOPIA_IMPORT_ERROR is not None:
        raise typer.BadParameter(
            "The official `sotopia` package is not installed or importable. "
            "Install it first, e.g. `pip install sotopia`, then initialize data with `sotopia install`. "
            f"Original import error: {_SOTOPIA_IMPORT_ERROR}"
        )

def _split_models(models: str) -> list[str]:
    return [item.strip() for item in models.split(",") if item.strip()]

@app.command("run")
def run_framework_benchmark(
    models: str = typer.Option("gpt-4o", help="Comma-separated test model names. The framework uses these as its model labels/backends."),
    partner_model: str = typer.Option("together_ai/meta-llama/Llama-3-70b-chat-hf", help="Official SOTOPIA partner model."),
    evaluator_model: str = typer.Option("gpt-4o", help="Official SOTOPIA evaluator/environment model."),
    batch_size: int = typer.Option(10, help="Official benchmark batch size."),
    task: str = typer.Option("hard", help="Official SOTOPIA task split, e.g. hard/all/cooperative/competitive."),
    url: str = typer.Option("", help="Optional official EnvAgentCombo JSON URL."),
    tag: str = typer.Option("framework_official_sotopia", help="Episode tag stored by official SOTOPIA."),
    push_to_db: bool = typer.Option(False, help="Whether official SOTOPIA should push results to DB."),
    output_to_jsonl: bool = typer.Option(False, help="Whether official SOTOPIA should export aggregated results."),
    save_dir: str = typer.Option(".", help="Directory for official JSONL output."),
    print_logs: bool = typer.Option(False, help="Show official SOTOPIA logs."),
) -> None:
    """Run the official SOTOPIA benchmark with this framework as the tested agent."""
    _ensure_sotopia_available()
    from sotopia.cli.benchmark.benchmark import _benchmark_impl

    _benchmark_impl(
        models=_split_models(models),
        agent_class=FrameworkSotopiaAgent,
        partner_model=partner_model,
        evaluator_model=evaluator_model,
        batch_size=batch_size,
        task=task,
        url=url,
        print_logs=print_logs,
        output_to_jsonl=output_to_jsonl,
        push_to_db=push_to_db,
        save_dir=save_dir,
        tag=tag,
    )

@app.command("display")
def display_framework_results(
    models: str = typer.Option("gpt-4o", help="Comma-separated model names to display."),
    partner_model: str = typer.Option("together_ai/meta-llama/Llama-3-70b-chat-hf", help="Official SOTOPIA partner model."),
    evaluator_model: str = typer.Option("gpt-4o", help="Evaluator model used for the stored episodes."),
    task: str = typer.Option("hard", help="Task split label used for display."),
    tag: str = typer.Option("framework_official_sotopia", help="Episode tag to aggregate."),
    output_to_jsonl: bool = typer.Option(False, help="Export official aggregated results."),
    save_dir: str = typer.Option(".", help="Directory for official JSONL output."),
) -> None:
    """Display/aggregate existing results using SOTOPIA's official benchmark_display."""
    _ensure_sotopia_available()
    from sotopia.cli.benchmark.benchmark import benchmark_display

    benchmark_display(
        model_list=_split_models(models),
        partner_model=partner_model,
        evaluator_model=evaluator_model,
        task=task,
        output_to_jsonl=output_to_jsonl,
        save_dir=save_dir,
        agent_class=FrameworkSotopiaAgent.__name__,
        tag=tag,
    )

if __name__ == "__main__":
    app()
