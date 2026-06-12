import logging
from typing import Optional, Dict, Any

from main import run_pipeline

logger = logging.getLogger(__name__)

class FrameworkBenchmarkAdapter:
    """
    Thin adapter that runs the uploaded cultural adaptive multi-agent framework
    on a single benchmark prompt while keeping the benchmark's evaluation logic
    unchanged.

    Each benchmark item is treated as an independent single-turn interaction,
    which preserves the original benchmark assumption that model.generate(prompt)
    is stateless across items.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        seed: int = 42,
        temperature_perception: float = 0.2,
        temperature_memory: float = 0.1,
        temperature_culture: float = 0.3,
        temperature_response: float = 0.7,
        n_hypotheses: int = 5,
        social_goal: str = "",
        agent_persona: str = "a culturally aware conversational assistant",
        debug: bool = False,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.seed = seed
        self.temperature_perception = temperature_perception
        self.temperature_memory = temperature_memory
        self.temperature_culture = temperature_culture
        self.temperature_response = temperature_response
        self.n_hypotheses = n_hypotheses
        self.social_goal = social_goal
        self.agent_persona = agent_persona
        self.debug = debug

    def generate(self, prompt: str) -> str:
        try:
            response, _memory, _cultural_state, _system_prompt = run_pipeline(
                current_input=prompt,
                prior_memory={},
                prior_cultural_state=None,
                timestep=0,
                base_url=self.base_url,
                api_key=self.api_key,
                model_name=self.model_name,
                temperature_perception=self.temperature_perception,
                temperature_memory=self.temperature_memory,
                temperature_culture=self.temperature_culture,
                temperature_response=self.temperature_response,
                seed=self.seed,
                n_hypotheses=self.n_hypotheses,
                debug=self.debug,
                social_goal=self.social_goal,
                agent_persona=self.agent_persona,
            )
            return response
        except Exception as e:
            logger.error("Framework adapter generation failed: %s", e, exc_info=True)
            return ""

    def close(self):
        pass
