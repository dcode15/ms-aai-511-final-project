from typing import Dict

import numpy as np
import pretty_midi


class MultidimensionalFeatureExtractor:
    @staticmethod
    def extract_features(midi_data: pretty_midi.PrettyMIDI, sampling_frequency: int = 100) -> Dict[str, np.ndarray]:
        return {
            "piano_roll": midi_data.get_piano_roll(sampling_frequency),
            "chroma_piano_roll": midi_data.get_chroma(sampling_frequency),
            "pitch_class_histogram": midi_data.get_pitch_class_histogram(),
            "pitch_class_transition_matrix": midi_data.get_pitch_class_transition_matrix(normalize=True),
        }
