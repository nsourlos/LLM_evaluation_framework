"""
Scoring function
"""

def verify_numerical_outputs(a, b, tol):
    """
    Compares two scalars, lists, or dictionaries of floats.
    Returns a score from 0 to 5 based on how many values match within an absolute tolerance.

    Parameters
    ----------
    a, b : float, list of floats, or dict of {str: float}
        Values to compare. If lists or dicts, they must be of equal length/keys.
    tol : float
        Absolute tolerance. Two numbers match if abs(a - b) <= tol.

    Returns
    -------
    float
        A score from 0 (no match) to 5 (perfect match), proportional to the fraction of values within tolerance.
    """
    def is_scalar(x):
        return isinstance(x, (float, int))

    def safe_float(x):
        try:
            return float(x)
        except Exception:
            raise ValueError(f"Cannot convert value {x} to float.")

    if is_scalar(a) and is_scalar(b):
        return 5.0 if abs(safe_float(a) - safe_float(b)) <= tol else 0.0

    elif isinstance(a, dict) and isinstance(b, dict):
        common_keys = a.keys() & b.keys()
        if not common_keys:
            raise ValueError("No common keys to compare.")
        matches = sum(
            abs(safe_float(a[k]) - safe_float(b[k])) <= tol for k in common_keys
        )
        return 5.0 * matches / len(common_keys)

    elif hasattr(a, '__len__') and hasattr(b, '__len__'):
        if len(a) != len(b):
            raise ValueError("Sequences must have the same length.")
        matches = sum(
            abs(safe_float(x) - safe_float(y)) <= tol for x, y in zip(a, b)
        )
        return 5.0 * matches / len(a)

    else:
        raise TypeError("Inputs must be scalars, sequences of floats, or dictionaries.") 