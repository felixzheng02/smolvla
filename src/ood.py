"""Out-of-distribution evaluation: paraphrased instructions and cross-suite transfer."""

from __future__ import annotations

import re

# Known LIBERO-10 (LIBERO-Long) instruction paraphrases.
# Each original instruction maps to semantically equivalent rephrasings.
LIBERO_10_PARAPHRASES: dict[str, list[str]] = {
    "put both the alphabet soup and the tomato sauce in the basket": [
        "place the soup can and the sauce container into the basket",
        "move the alphabet soup and tomato sauce into the basket",
    ],
    "put both the cream cheese box and the butter in the basket": [
        "move the cream cheese and the butter into the basket",
        "place the cream cheese box and butter inside the basket",
    ],
    "turn on the stove and put the moka pot on it": [
        "switch on the burner and set the coffee pot on top",
        "activate the stove and place the moka pot on it",
    ],
    "put the black bowl in the bottom drawer of the cabinet and close it": [
        "place the dark bowl into the lower cabinet drawer and shut it",
        "set the black bowl in the bottom drawer then close the drawer",
    ],
    "put the white mug on the left plate and put the yellow and white mug on the right plate": [
        "set the white cup on the left dish and the striped mug on the right dish",
        "place the white mug on the left plate and the two-tone mug on the right plate",
    ],
    "pick up the book and place it in the back compartment of the caddy": [
        "grab the book and put it in the rear section of the caddy",
        "take the book and set it in the back part of the caddy",
    ],
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": [
        "set the white cup on the dish and move the pudding to the plate's right side",
        "place the white mug on the plate then put the chocolate pudding on its right",
    ],
    "put both the alphabet soup and the cream cheese box in the basket": [
        "place the soup can and cream cheese into the basket",
        "move the alphabet soup and the cream cheese box into the basket",
    ],
    "put both moka pots on the stove": [
        "set both coffee pots on the stovetop",
        "place the two moka pots onto the stove",
    ],
    "put the yellow and white mug in the microwave and close it": [
        "place the striped mug inside the microwave and shut the door",
        "set the two-tone mug in the microwave then close it",
    ],
}

# Legacy paraphrases for other LIBERO suites.
LIBERO_PARAPHRASES: dict[str, list[str]] = {
    **LIBERO_10_PARAPHRASES,
    "pick up the black bowl on the left and place it on the plate": [
        "grasp the left black bowl and set it onto the plate",
    ],
    "push the plate to the front of the stove": [
        "slide the plate toward the front of the stove",
    ],
    "put the cream cheese in the bowl": [
        "place the cream cheese into the bowl",
    ],
    "turn on the stove": [
        "switch the stove on",
    ],
    "put the bowl on the plate": [
        "place the bowl onto the plate",
    ],
    "put the wine bottle on top of the cabinet": [
        "place the wine bottle on the cabinet top",
    ],
    "open the top drawer and put the bowl inside": [
        "pull open the top drawer then place the bowl in it",
    ],
    "put the bowl on the stove": [
        "place the bowl onto the stove",
    ],
    "pick up the book and place it in the back compartment of the caddy": [
        "grab the book and put it into the caddy's rear compartment",
    ],
    "pick up the alphabet soup and place it in the basket": [
        "grab the alphabet soup and drop it into the basket",
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


def get_paraphrase_map(task_suite: str = "libero_10", variant: int = 0) -> dict[str, str]:
    """Return a mapping of original -> paraphrased instruction for all tasks in a suite.

    Args:
        task_suite: LIBERO suite name.
        variant: Which paraphrase variant to use (0 = first, 1 = second, etc.).

    Returns:
        Dict mapping original instruction string to its paraphrase.
    """
    source = LIBERO_10_PARAPHRASES if "10" in task_suite or "long" in task_suite.lower() else LIBERO_PARAPHRASES
    result = {}
    for original, paraphrases in source.items():
        idx = min(variant, len(paraphrases) - 1)
        result[original] = paraphrases[idx]
    return result


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
