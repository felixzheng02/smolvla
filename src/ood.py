"""Out-of-distribution evaluation: paraphrased instructions and cross-suite transfer."""

from __future__ import annotations

import re

# Known LIBERO-10 instruction paraphrases.
LIBERO_PARAPHRASES: dict[str, list[str]] = {
    "pick up the black bowl on the left and place it on the plate": [
        "grasp the left black bowl and set it onto the plate",
        "take the black bowl from the left side and put it on the plate",
        "lift the black bowl on the left and drop it on the plate",
    ],
    "push the plate to the front of the stove": [
        "slide the plate toward the front of the stove",
        "move the plate forward to the stove's front edge",
    ],
    "put the cream cheese in the bowl": [
        "place the cream cheese into the bowl",
        "drop the cream cheese inside the bowl",
    ],
    "turn on the stove": [
        "switch the stove on",
        "activate the stove burner",
    ],
    "put the bowl on the plate": [
        "place the bowl onto the plate",
        "set the bowl down on the plate",
    ],
    "put the wine bottle on top of the cabinet": [
        "place the wine bottle on the cabinet top",
        "set the wine bottle onto the top of the cabinet",
    ],
    "open the top drawer and put the bowl inside": [
        "pull open the top drawer then place the bowl in it",
        "open the upper drawer and set the bowl inside",
    ],
    "put the bowl on the stove": [
        "place the bowl onto the stove",
        "set the bowl on the stove top",
    ],
    "pick up the book and place it in the back compartment of the caddy": [
        "grab the book and put it into the caddy's rear compartment",
        "take the book and set it in the back section of the caddy",
    ],
    "pick up the alphabet soup and place it in the basket": [
        "grab the alphabet soup and drop it into the basket",
        "take the alphabet soup and put it in the basket",
    ],
}

# Synonym table for fallback paraphrasing.
_SYNONYMS: dict[str, list[str]] = {
    "pick up": ["grasp", "grab", "take"],
    "place": ["put", "set", "drop"],
    "push": ["slide", "move"],
    "put": ["place", "set"],
    "open": ["pull open"],
    "turn on": ["switch on", "activate"],
    "lift": ["raise", "pick up"],
}


def _fallback_paraphrase(instruction: str) -> list[str]:
    """Generate simple paraphrases via synonym replacement for unknown instructions."""
    results: list[str] = []
    lower = instruction.lower().strip()
    for phrase, replacements in _SYNONYMS.items():
        if phrase in lower:
            for repl in replacements[:2]:
                variant = re.sub(re.escape(phrase), repl, lower, count=1)
                if variant != lower:
                    results.append(variant)
            break  # apply first matching synonym group only
    return results if results else [lower]


def paraphrase_instruction(instruction: str) -> list[str]:
    """Return paraphrased versions of a LIBERO instruction.

    Uses a lookup dict for known instructions, falls back to simple
    synonym-based string transformations for unknown ones.
    """
    key = instruction.lower().strip()
    if key in LIBERO_PARAPHRASES:
        return list(LIBERO_PARAPHRASES[key])
    return _fallback_paraphrase(instruction)


# Cross-suite evaluation configurations.
_CROSS_SUITE_CONFIGS: dict[str, dict[str, str]] = {
    "libero_10": {
        "repo_id": "lerobot/libero_10",
        "env_type": "libero",
        "suite": "libero_10",
    },
    "libero_spatial": {
        "repo_id": "lerobot/libero_spatial",
        "env_type": "libero",
        "suite": "libero_spatial",
    },
    "libero_object": {
        "repo_id": "lerobot/libero_object",
        "env_type": "libero",
        "suite": "libero_object",
    },
    "libero_goal": {
        "repo_id": "lerobot/libero_goal",
        "env_type": "libero",
        "suite": "libero_goal",
    },
}

# Default transfer targets: train on source -> eval on these suites.
_TRANSFER_MAP: dict[str, list[str]] = {
    "libero_10": ["libero_spatial", "libero_object", "libero_goal"],
    "libero_spatial": ["libero_10", "libero_object"],
    "libero_object": ["libero_10", "libero_spatial"],
    "libero_goal": ["libero_10", "libero_spatial"],
}


def get_cross_suite_config(source_suite: str) -> dict[str, object]:
    """Return eval config for cross-suite transfer from *source_suite*.

    Returns a dict with 'source' info and a list of 'targets', each
    containing repo_id, env_type, and suite name.
    """
    source_key = source_suite.lower().strip()
    if source_key not in _CROSS_SUITE_CONFIGS:
        raise ValueError(
            f"Unknown suite '{source_suite}'. "
            f"Available: {sorted(_CROSS_SUITE_CONFIGS)}"
        )
    target_keys = _TRANSFER_MAP.get(source_key, [])
    return {
        "source": _CROSS_SUITE_CONFIGS[source_key],
        "targets": [_CROSS_SUITE_CONFIGS[t] for t in target_keys],
    }
