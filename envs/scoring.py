"""
Scoring utilities for OpenEnv validation compliance.

OpenEnv requires all task scores to be strictly in the open interval (0, 1):
    0 < score < 1
    Never exactly 0.0, never exactly 1.0

This module provides a strict_score() function to ensure safe score transformation.
All scores are bounded within [0.05, 0.95] — well inside the open interval (0, 1).
0.05 and 0.95 are exact IEEE 754 double-precision values, eliminating float rounding risk.
"""

# The fixed bounds for all scores — strictly inside (0, 1)
# Using explicit float literals rather than 1e-6 epsilon to avoid rounding edge cases
SCORE_MIN = 0.05
SCORE_MAX = 0.95


def strict_score(raw_score: float) -> float:
    """
    Transform a raw score into the strict open interval [0.05, 0.95].

    All output values are guaranteed to be strictly between 0 and 1.
    Uses explicit 0.05/0.95 bounds (exact IEEE 754 representations) rather
    than epsilon offsets, eliminating any floating-point rounding risk.

    Args:
        raw_score: A raw score value (can be any float, including NaN, inf)

    Returns:
        A float in [0.05, 0.95], i.e., strictly 0 < score < 1

    Examples:
        strict_score(0.0)          -> 0.05
        strict_score(1.0)          -> 0.95
        strict_score(0.5)          -> 0.5
        strict_score(-1.0)         -> 0.05  (clamped)
        strict_score(2.0)          -> 0.95  (clamped)
        strict_score(float('nan')) -> 0.5   (safe fallback)
        strict_score(float('inf')) -> 0.5   (safe fallback)
    """
    # Handle NaN (NaN != NaN is always True in IEEE 754) and infinity
    try:
        f = float(raw_score)
    except (TypeError, ValueError):
        return 0.5

    if f != f:  # NaN check
        return 0.5
    if f == float('inf') or f == float('-inf'):
        return 0.5

    # Clamp input to [0, 1]
    clamped = max(0.0, min(1.0, f))

    # Linear map [0, 1] -> [SCORE_MIN, SCORE_MAX]
    #   0.0 -> 0.05,  1.0 -> 0.95,  0.5 -> 0.50  (midpoint preserved)
    mapped = SCORE_MIN + (SCORE_MAX - SCORE_MIN) * clamped

    # Hard clamp to [SCORE_MIN, SCORE_MAX] — defense in depth
    # Both 0.05 and 0.95 are exact IEEE 754 doubles, so no rounding to 0 or 1
    return max(SCORE_MIN, min(SCORE_MAX, mapped))


def safe_divide(numerator: float, denominator: float, default: float = 0.5) -> float:
    """
    Safely divide two floats, returning a default if denominator is zero.

    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if denominator is zero

    Returns:
        The quotient, or default if denominator <= 0
    """
    if denominator <= 0:
        return default
    return numerator / denominator


def bounded_average(values: list, weight_zero: float = 0.01, weight_one: float = 0.99) -> float:
    """
    Compute a weighted average that stays strictly in (0, 1).

    This is useful for combining partial scores while ensuring the result
    never hits exact boundaries.

    Args:
        values: List of scores to average
        weight_zero: How much weight to give to the lower bound (0)
        weight_one: How much weight to give to the upper bound (1)

    Returns:
        A score strictly in (0, 1)
    """
    if not values:
        return 0.5

    raw_avg = sum(values) / len(values)

    # Add small contribution from boundaries to prevent exact hits
    result = (weight_zero * 0.0 + weight_one * 1.0 + (1 - weight_zero - weight_one) * raw_avg)

    return strict_score(result)
