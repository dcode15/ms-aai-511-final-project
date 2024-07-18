from typing import Dict, List

import numpy as np
import pretty_midi


class PitchFeatureExtractor:

    @staticmethod
    def extract_features(midi_data: pretty_midi.PrettyMIDI) -> Dict[str, float]:
        """
        Extracts pitch-related features from the given MIDI data.

        Args:
            midi_data (pretty_midi.PrettyMIDI): The MIDI data to analyze.

        Returns:
            Dict[str, float]: A dictionary containing the extracted features:
                - pitch_variety: The number of unique pitches used.
                - pitch_range: The range between the highest and lowest pitches.
                - primary_register: The mean pitch value.
                - importance_bass_register: The proportion of notes in the bass register (0-54).
                - importance_middle_register: The proportion of notes in the middle register (55-72).
                - importance_high_register: The proportion of notes in the high register (73-127).
        """
        notes: List[pretty_midi.Note] = [note for instrument in midi_data.instruments for note in instrument.notes]
        pitches: np.ndarray = np.array([note.pitch for note in notes])

        # Pitches
        pitch_variety: int = len(np.unique(pitches))
        pitch_range: int = np.max(pitches) - np.min(pitches)

        # Registers
        primary_register: float = np.mean(pitches)
        total_notes: int = len(notes)
        bass_notes: int = np.sum((pitches >= 0) & (pitches <= 54))
        middle_notes: int = np.sum((pitches >= 55) & (pitches <= 72))
        high_notes: int = np.sum((pitches >= 73) & (pitches <= 127))

        importance_bass: float = bass_notes / total_notes
        importance_middle: float = middle_notes / total_notes
        importance_high: float = high_notes / total_notes

        return {
            "pitch_variety": pitch_variety,
            "pitch_range": pitch_range,
            "primary_register": primary_register,
            "importance_bass_register": importance_bass,
            "importance_middle_register": importance_middle,
            "importance_high_register": importance_high
        }
