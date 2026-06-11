"""
Module: ssm_data_maker
Description: Extracts Bach chorales from the music21 corpus and converts them 
into structured NumPy arrays representing MIDI pitches over time.
"""
from typing import List

import numpy as np
from music21 import corpus, note
from tqdm import tqdm


def process_chorales_to_grids(step: float = 0.25, max_parts: int = 4) -> List[np.ndarray]:
    """
    Extracts chorales from the music21 corpus into time-quantized grids.

    Args:
        step (float): The quantization step in quarter lengths. Defaults to 0.25.
        max_parts (int): Maximum number of voices to extract. Defaults to 4.

    Returns:
        List[np.ndarray]: A list of arrays, each of shape (time_steps, max_parts).
    """
    chorales = corpus.chorales.Iterator()
    all_songs = []

    for chorale in tqdm(chorales, desc="Processing Chorales"):
        parts = list(chorale.parts)[:max_parts]
        max_time = 0
        events = []

        # 1. Compute global max time and extract events
        for p_idx, part in enumerate(parts):
            for n in part.flatten().notes:
                pitch = n.pitch.midi if isinstance(n, note.Note) else n.root().midi
                start = int(n.offset / step)
                dur = int(n.duration.quarterLength / step)

                max_time = max(max_time, start + dur)
                events.append((p_idx, start, dur, pitch))

        # 2. Create and fill the time/pitch grid
        song = np.zeros((max_time + 1, max_parts), dtype=int)
        for v_idx, start, dur, pitch in events:
            song[start:start+dur, v_idx] = pitch
            
        all_songs.append(song)

    return all_songs