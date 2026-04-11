"""
Scoring utilities for OpenEnv validation compliance.

OpenEnv requires all task scores to be strictly in the open interval (0, 1):
    0 < score < 1
    Never exactly 0.0, never exactly 1.0

This module provides a strict_score() function to ensure safe score transformation.
"""


# ── Open-interval score bounds (used across the entire codebase) ──────────────
# All grader scores must satisfy: SCORE_MIN < score < SCORE_MAX
# These values provide a safe buffer inside the strict (0, 1) requirement.
SCORE_MIN: float = 0.05
SCORE_MAX: float = 0.95


def strict_score(raw_score: float) -> float:
    """
    Transform a raw score into the strict open interval (0, 1).
    
    This function handles all edge cases and ensures that no score can ever be
    exactly 0.0 or 1.0, regardless of input.
    
    Args:
        raw_score: A raw score value (can be any float, including NaN, inf)
    
    Returns:
        A float strictly in (0, 1), i.e., 0 < score < 1
    
    Examples:
        strict_score(0.0)     -> 0.05
        strict_score(1.0)     -> 0.95
        strict_score(0.5)     -> 0.5
        strict_score(-1.0)    -> 0.05 (fallback)
        strict_score(2.0)     -> 0.95 (fallback)
        strict_score(float('nan'))  -> 0.5 (safe fallback)
    """
    # Handle NaN or inf
    if raw_score != raw_score or abs(raw_score) == float('inf'):
        return 0.5  # Safe fallback for invalid inputs
    
    # Clamp to [0, 1]
    clamped = max(0.0, min(1.0, raw_score))
    
    # Map [0, 1] to (0.05, 0.95) to ensure strict openness
    # This means:
    #   0.0 -> 0.05
    #   1.0 -> 0.95
    #   0.5 -> 0.5
    mapped = 0.05 + 0.90 * clamped
    
    # Final safety check (should be redundant but defensive)
    return max(1e-6, min(1.0 - 1e-6, mapped))


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


def bounded_average(values: list[float], weight_zero=0.01, weight_one=0.99) -> float:
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
