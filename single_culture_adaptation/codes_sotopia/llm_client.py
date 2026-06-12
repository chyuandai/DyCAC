import logging
import random
import time

from openai import OpenAI

_logger = logging.getLogger("llm_client")


def call_llm(
    system_prompt: str,
    user_message: str,
    base_url: str = "https://api.openai.com/v1",
    api_key: str = None,          # None → SDK auto-reads OPENAI_API_KEY env var
    model_name: str = "gpt-4o",
    temperature: float = 0.3,
    seed: int = 42,
    max_tokens: int = 2048,
    _module: str = "",             # optional telemetry label, e.g. "perception"
    # Retry config — handles rate-limit (429) and transient server errors (5xx)
    max_retries: int = 4,
    base_delay: float = 2.0,       # seconds; actual delay = base_delay * 2^(attempt-1) + jitter
) -> str:
    """
    Call an OpenAI-compatible LLM and return the assistant's text response.

    Parameters
    ----------
    system_prompt : str
        The system-level instruction prompt.
    user_message : str
        The user-turn message to send.
    base_url : str
        Base URL for the API endpoint (supports OpenAI, Azure, local models, etc.).
    api_key : str or None
        API key for authentication. If None (default), the OpenAI SDK automatically
        reads the OPENAI_API_KEY environment variable.
    model_name : str
        Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022').
    temperature : float
        Sampling temperature (0.0 = deterministic, 1.0 = creative).
    seed : int
        Random seed for reproducibility (supported by some providers).
    max_tokens : int
        Maximum number of tokens in the response.
    _module : str
        Optional label attached to the telemetry record for this call (e.g.
        ``"perception"``, ``"memory"``, ``"culture"``, ``"planning"``).
        Has no effect if no telemetry collector is active.

    Returns
    -------
    str
        The assistant's response text.

    Raises
    ------
    RuntimeError
        If the API call fails or returns an unexpected response.
    """
    effective_api_key = api_key if api_key else None
    client = OpenAI(api_key=effective_api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    completion = None
    t0 = time.perf_counter()
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
            )
            break   # success — exit retry loop

        except Exception as e:
            err_str = str(e).lower()
            # Detect retryable conditions: rate-limit (429) or transient server errors (5xx)
            is_rate_limit   = "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str
            is_server_error = any(code in err_str for code in ("500", "502", "503", "504"))
            is_retryable    = is_rate_limit or is_server_error

            if attempt < max_retries and is_retryable:
                # Exponential back-off with jitter to avoid thundering-herd
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.5, 1.5)
                _logger.warning(
                    "[LLMClient] %s — attempt %d/%d failed (%s). Retrying in %.1fs…",
                    _module or model_name, attempt, max_retries,
                    "rate-limit" if is_rate_limit else "server-error",
                    delay,
                )
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"[LLMClient] API call failed after {attempt} attempt(s): {e}"
                ) from e

    latency_s = time.perf_counter() - t0
    response  = completion.choices[0].message.content.strip()

    # ── Report to the active telemetry collector (if any) ─────────────────
    # This is a zero-cost no-op when no collector is registered.
    try:
        from telemetry import get_active_collector, CallStats
        collector = get_active_collector()
        if collector is not None and completion.usage is not None:
            usage = completion.usage
            collector.record(CallStats(
                prompt_tokens     = getattr(usage, "prompt_tokens",     0),
                completion_tokens = getattr(usage, "completion_tokens", 0),
                total_tokens      = getattr(usage, "total_tokens",      0),
                latency_s         = latency_s,
                module            = _module,
            ))
    except ImportError:
        pass   # telemetry.py not present → silently skip

    return response