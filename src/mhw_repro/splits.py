"""Chronological data partitions used to avoid temporal leakage."""


def chronological_base_meta_split(number_of_rows, base_fraction=0.75, minimum_each=1):
    """Return non-overlapping slices for base-model and meta-model fitting."""
    if number_of_rows < 2 * minimum_each:
        raise ValueError("Not enough rows for the requested chronological split")
    if not 0 < base_fraction < 1:
        raise ValueError("base_fraction must be between zero and one")
    cut = int(number_of_rows * base_fraction)
    cut = max(minimum_each, min(cut, number_of_rows - minimum_each))
    return slice(0, cut), slice(cut, number_of_rows)

