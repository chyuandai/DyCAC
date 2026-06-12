"""
Perception Ablation Stub

"""

from __future__ import annotations

from typing import Optional

def build_perception_stub(
    current_input: str,
    prior_memory: Optional[dict] = None,
    social_goal: str = "",
) -> dict:
    """
    Return a schema-compatible empty perception object.

    This preserves the downstream contracts expected by:
      - memory_update.run_memory_update()
      - cultural_hypothesis.run_cultural_hypothesis()
      - planning_execution.run_planning_execution()

    No inference is performed here. This is a pure ablation stub.
    """
    perception = {
        "objective_facts": {
            "static_attributes": {},
            "dynamic_events": [],
        },
        "mental_state": {
            "immediate_intent": "",
            "emotion": {
                "category": "neutral",
                "intensity": "low",
            },
            "values_and_obsessions": [],
        },
        "cultural_cues": {
            "communication_style": "",
            "social_tendencies": [],
        },
    }

    if social_goal:
        perception["goal_tracking"] = {
            "progress_assessment": "",
            "partner_signals": [],
            "obstacles": [],
            "leverage_points": [],
            "suggested_next_move": "",
            "goal_stage": "",
            "partner_stance": "",
            "risk_flags": [],
        }

    return perception
