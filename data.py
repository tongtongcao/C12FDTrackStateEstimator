import numpy as np


def read_tracks_with_hits(filename):
    """
    Read a CSV file where each event (track) is represented by two lines:

      Line 1:
        A flattened list of hit-level features. Each hit consists of
        five values: (doca, xm, xr, yr, z).
        The total number of values on this line must be a multiple of 5.

      Line 2:
        The corresponding initial track state at z = 229
        in the tilted sector frame:
        (x, y, tx, ty, q/p)

    Parameters
    ----------
    filename : str
        Path to the input CSV file.

    Returns
    -------
    hits_list : list of np.ndarray
        List of hit arrays, one per track.
        Each array has shape [num_hits, 5].

    states : np.ndarray
        Array of track state vectors with shape [N, 5],
        where N is the number of tracks.
    """
    hits_list = []
    states = []

    # Read file and ignore empty lines
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != ""]

    # Each event must consist of exactly two lines
    if len(lines) % 2 != 0:
        raise ValueError("The file must contain an even number of lines (two lines per event).")

    # Loop over events (two lines at a time)
    for i in range(0, len(lines), 2):
        # --------------------------------------------------
        # First line: hit-level information
        hit_values = [float(x) for x in lines[i].split(",")]

        # The number of hit values must be divisible by 5
        if len(hit_values) % 5 != 0:
            raise ValueError(
                f"Line {i + 1}: number of hit values ({len(hit_values)}) is not a multiple of 5"
            )

        # Reshape into [num_hits, 5]
        hits = np.array(hit_values, dtype=np.float32).reshape(-1, 5)
        hits_list.append(hits)

        # --------------------------------------------------
        # Second line: track initial state
        state_values = [float(x) for x in lines[i + 1].split(",")]

        if len(state_values) != 5:
            raise ValueError(
                f"Line {i + 2}: track state must contain exactly 5 values, got {len(state_values)}"
            )

        states.append(state_values)

    states = np.array(states, dtype=np.float32)  # [N, 5]
    return hits_list, states


# --------------------------------------------------
# Example usage
if __name__ == "__main__":
    hits, states = read_tracks_with_hits("sample.csv")

    print(f"Read {len(hits)} tracks")
    print(f"First track hits shape: {hits[0].shape}")   # (num_hits, 5)
    print(f"First track initial state: {states[0]}")
