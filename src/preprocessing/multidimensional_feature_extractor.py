from typing import Dict

import numpy as np
import pretty_midi


class MultidimensionalFeatureExtractor:

    @staticmethod
    def extract_features(midi_data: pretty_midi.PrettyMIDI, sampling_frequency: int = 10) -> Dict[str, np.ndarray]:
        """
        Extracts multidimensional features from the given MIDI data.

        Args:
            midi_data (pretty_midi.PrettyMIDI): The MIDI data to analyze.
            sampling_frequency (int, optional): The sampling frequency for piano roll and chroma features. Defaults to 10 Hz.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the extracted features:
                - piano_roll: A 2D numpy array representing the piano roll of the MIDI data.
                - chroma_piano_roll: A 2D numpy array representing the chromagram of the MIDI data.
                - pitch_class_histogram: A 1D numpy array representing the distribution of pitch classes.
                - pitch_class_transition_matrix: A 2D numpy array representing the normalized transition probabilities between pitch classes.
        """
        return {
            "piano_roll": midi_data.get_piano_roll(sampling_frequency),
            "chroma_piano_roll": midi_data.get_chroma(sampling_frequency),
            "pitch_class_histogram": midi_data.get_pitch_class_histogram(),
            "pitch_class_transition_matrix": midi_data.get_pitch_class_transition_matrix(normalize=True),
        }
