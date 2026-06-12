"""
Main Entry Point — Ablated Adaptive Multi-Agent Framework
----------------------------------------------------------
Orchestrates the ablated three-module pipeline for single-turn or multi-turn dialogue.

Usage:
    python main.py --input "Hello, I am Li Wei, I just failed a job interview yesterday." \
                   --api_key "$OPENAI_API_KEY" \
                   --model_name gpt-4o \
                   --debug

Multi-turn interactive mode (omit --input):
    python main.py --api_key "$OPENAI_API_KEY" --model_name gpt-4o --debug
"""

import argparse
import json
import sys
from typing import Optional

from perception import run_perception
from memory_update import run_memory_update
from planning_execution import run_planning_execution

def ingest_input(
    current_input: str,
    prior_memory: Optional[dict] = None,
    timestep: int = 0,
) -> tuple[str, Optional[dict]]:
    """
    Unified data ingestion interface.

    This function serves as the entry point for both:
    - First-turn input (t=0): only current dialogue text, no prior memory.
    - Subsequent turns  (t>0): current dialogue text + prior memory dict.

    The input text may represent:
    - A single conversational utterance
    - A multi-turn dialogue block
    - A story/narrative slice

    Parameters
    ----------
    current_input : str
        The dialogue or narrative text at the current timestep.
    prior_memory : dict, optional
        Memory dictionary from the previous timestep. None or {} if first turn.
    timestep : int
        Current timestep index (0 = first turn).

    Returns
    -------
    tuple[str, Optional[dict]]
        (current_input, prior_memory) — passed through for downstream processing.
        Concrete parsing logic should be implemented here as needed.
    """

    return current_input, prior_memory if prior_memory else {}

def run_pipeline(
    current_input: str,
    prior_memory: dict,
    prior_cultural_state: Optional[dict],
    timestep: int,

    base_url: str,
    api_key: str,
    model_name: str,
    temperature_perception: float,
    temperature_memory: float,
    temperature_culture: float,
    temperature_response: float,
    seed: int,
    n_hypotheses: int,
    debug: bool,

    social_goal: str = "",
    agent_persona: str = "a context-aware conversational assistant",
) -> tuple[str, dict, dict, str]:
    """
    Run one full turn of the Ablated Adaptive Multi-Agent Framework.

    Returns
    -------
    tuple[str, dict, dict, str]
        (response, updated_memory, updated_cultural_state, system_prompt)
    """
    llm_kwargs = dict(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        seed=seed,
        debug=debug,
    )

    dialogue_meta = prior_memory.get("dialogue_meta", {}) if prior_memory else {}
    recent_turns = dialogue_meta.get("recent_turns", []) if isinstance(dialogue_meta, dict) else []
    if recent_turns:
        history_block = "\n".join(
            f"{t.get('speaker', '')}: {t.get('utterance', '')}" for t in recent_turns[-6:] if t.get('utterance')
        )
        analysis_input = (
            f"[Conversation context]\nRecent dialogue:\n{history_block}\n\n"
            f"[Latest message]\n{current_input}\n\n"
            f"Task: Use prior turns only as context. Analyze and respond to the latest message only."
        )
    else:
        analysis_input = current_input

    perception = run_perception(
        current_input=analysis_input,
        prior_memory=prior_memory,
        temperature=temperature_perception,
        social_goal=social_goal,
        **llm_kwargs,
    )

    updated_memory = run_memory_update(
        prior_memory=prior_memory,
        perception=perception,
        temperature=temperature_memory,
        **llm_kwargs,
    )
    updated_cultural_state = None

    goal_tracking = perception.get("goal_tracking") if social_goal else None
    system_prompt, response = run_planning_execution(
        memory=updated_memory,
        current_input=analysis_input,
        temperature=temperature_response,
        social_goal=social_goal,
        agent_persona=agent_persona,
        goal_tracking=goal_tracking,
        **llm_kwargs,
    )

    if isinstance(updated_memory, dict):
        meta = updated_memory.setdefault("dialogue_meta", {})
        recent = list(meta.get("recent_turns", []) or [])
        recent.append({"speaker": "Partner", "utterance": current_input})
        recent.append({"speaker": "Self", "utterance": response})
        meta["recent_turns"] = recent[-8:]

    return response, updated_memory, updated_cultural_state, system_prompt

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablated Adaptive Multi-Agent Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--social_goal", type=str, default="",
        help=(
            "Optional social goal for the agent (SOTOPIA-style). "
            "E.g. 'Convince your friend to lend you their car for the weekend.' "
            "When set, the agent pursues this goal using context-aware dialogue."
        ),
    )
    parser.add_argument(
        "--agent_persona", type=str, default="a context-aware conversational assistant",
        help="Brief role description for the agent (used when --social_goal is set).",
    )

    parser.add_argument(
        "--input", type=str, default=None,
        help="Single dialogue input text. If omitted, launches interactive multi-turn mode."
    )
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Path to a JSON file containing a list of dialogue turns (list of strings)."
    )

    parser.add_argument("--base_url", type=str, default="",
                        help="LLM API base URL.")
    parser.add_argument("--api_key", type=str, default="",
                        help="LLM API key.")
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        help="Model identifier.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    parser.add_argument("--temp_perception", type=float, default=0.2,
                        help="Temperature for the Perception module.")
    parser.add_argument("--temp_memory", type=float, default=0.1,
                        help="Temperature for the Memory Update module.")
    parser.add_argument("--temp_culture", type=float, default=0.0,
                        help="Deprecated/ignored: Cultural Hypothesis is ablated.")
    parser.add_argument("--temp_response", type=float, default=0.7,
                        help="Temperature for the Planning & Execution module.")

    parser.add_argument("--n_hypotheses", type=int, default=0,
                        help="Deprecated/ignored: Cultural Hypothesis is ablated.")

    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode: print all module inputs and outputs.")

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    pipeline_kwargs = dict(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        temperature_perception=args.temp_perception,
        temperature_memory=args.temp_memory,
        temperature_culture=args.temp_culture,
        temperature_response=args.temp_response,
        seed=args.seed,
        n_hypotheses=args.n_hypotheses,
        debug=args.debug,
        social_goal=args.social_goal,
        agent_persona=args.agent_persona,
    )

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            turns = json.load(f)
        if not isinstance(turns, list):
            print("[ERROR] --input_file must contain a JSON array of strings.", file=sys.stderr)
            sys.exit(1)
    elif args.input:
        turns = [args.input]
    else:
        turns = None

    memory: dict = {}
    cultural_state: Optional[dict] = None

    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _CYAN   = "\033[36m"
    _GREEN  = "\033[32m"
    _YELLOW = "\033[33m"

    def _c(text, *codes):
        return "".join(codes) + str(text) + _RESET

    if turns is not None:

        for t, turn_input in enumerate(turns):
            print(f"\n{_c('━'*62, _CYAN)}")
            print(f"  {_c(f'Turn {t}', _BOLD, _CYAN)}  │  {_c(turn_input[:80], _YELLOW)}")
            print(_c('━'*62, _CYAN))

            current_input, current_memory = ingest_input(turn_input, memory, timestep=t)
            response, memory, cultural_state, _ = run_pipeline(
                current_input=current_input,
                prior_memory=current_memory,
                prior_cultural_state=cultural_state,
                timestep=t,
                **pipeline_kwargs,
            )
            print(f"\n{_c('◆ ASSISTANT', _GREEN, _BOLD)}: {response}")
        return

    print(_c("\nAblated Adaptive Multi-Agent Framework", _BOLD, _CYAN))
    print("Type 'exit' or 'quit' to stop.\n")
    t = 0
    while True:
        try:
            user_input = input(_c("You: ", _YELLOW))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not user_input.strip():
            continue

        current_input, current_memory = ingest_input(user_input, memory, timestep=t)
        try:
            response, memory, cultural_state, _ = run_pipeline(
                current_input=current_input,
                prior_memory=current_memory,
                prior_cultural_state=cultural_state,
                timestep=t,
                **pipeline_kwargs,
            )
        except Exception as e:
            print(f"[ERROR] Pipeline failed at turn {t}: {e}", file=sys.stderr)
            if args.debug:
                import traceback
                traceback.print_exc()
            continue

        print(f"\n{_c('◆ Assistant', _GREEN, _BOLD)}: {response}\n")
        t += 1

if __name__ == "__main__":
    main()
