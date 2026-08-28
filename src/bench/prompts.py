"""Deterministic synthetic prompts of controlled length.

Prompt length is held fixed across a sweep so that changes in throughput come
from concurrency and serving config, not from input size drift. Prompts are
distinct from each other to defeat prefix caching, which would otherwise make
TTFT look far better than it would be in production.
"""

import random

_WORDS = (
    "system network latency cluster tensor kernel memory scheduler batch token "
    "gradient throughput pipeline cache register warp buffer stream device host "
    "matrix vector inference decode prefill attention weight layer embedding"
).split()


def build_prompts(n: int, target_tokens: int, seed: int = 0) -> list[str]:
    """Build n prompts each roughly target_tokens long.

    Approximation: ~0.75 words per token for this vocabulary. Exact token counts
    are reported by the server in the usage block, so this only needs to be close.
    """
    rng = random.Random(seed)
    n_words = max(1, int(target_tokens * 0.75))
    prompts = []
    for i in range(n):
        body = " ".join(rng.choice(_WORDS) for _ in range(n_words))
        # Unique prefix per prompt prevents cross-request prefix cache hits.
        prompts.append(f"[req-{i:05d}] {body}\nSummarize the above:")
    return prompts
