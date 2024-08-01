import os
import random
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm

from src.preprocessing.feature_extractor import FeatureExtractor


class DataAugmentor:

    @staticmethod
    def generate_augmented_songs(data_directory: str, training_data: pd.DataFrame, song_count: int) -> Tuple[pd.DataFrame, List[Dict[str, np.ndarray]]]:
        """
        Generates augmented songs based on the provided training data.

        Args:
            data_directory (str): The directory containing the MIDI files.
            training_data (pd.DataFrame): A DataFrame containing 'file_name' and 'composer' columns.
            song_count (int): The number of augmented songs to generate.

        Returns:
            Tuple[pd.DataFrame, List[Dict[str, np.ndarray]]]: A tuple containing:
                - A DataFrame of scalar features for the augmented songs.
                - A list of dictionaries containing multidimensional features for the augmented songs.
        """

        all_scalar_features = []
        all_multidimensional_features = []

        for _ in tqdm(range(song_count), desc="Generating Augmented Songs", unit="song"):
            sampled_file = training_data.sample()
            sampled_file_name = sampled_file.iloc[0]['file_name']
            sampled_file_composer = sampled_file.iloc[0]['composer']
            file_path = os.path.join(data_directory, sampled_file_composer,
                                     sampled_file_name)
            midi_data = DataAugmentor._augment_midi(file_path)
            scalar_features, multidimensional_features = FeatureExtractor.extract_features(midi_data)
            all_scalar_features.append(scalar_features)
            all_multidimensional_features.append(multidimensional_features)

        return pd.DataFrame(all_scalar_features), all_multidimensional_features

    @staticmethod
    def _augment_midi(input_file) -> pretty_midi.PrettyMIDI:
        """
        Augments a MIDI file by applying the following augmentations:
        1. Pitch shifting: Shifts the pitch by a random amount between -2 and 2 semitones.
        2. Time stretching: Stretches or compresses the timing of notes by a random factor between 0.95 and 1.05.
        3. Note density alteration: Randomly adds or removes notes to change the note density.

        Args:
            input_file (str): The path to the input MIDI file.

        Returns:
            pretty_midi.PrettyMIDI: The augmented MIDI data.
        """
        midi_data = pretty_midi.PrettyMIDI(input_file)

        # 1. Pitch shifting
        semitones = random.uniform(-2, 2)
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    note.pitch += int(semitones)
                    note.pitch = max(0, min(127, note.pitch))

        # 2. Time stretching
        stretch_factor = random.uniform(0.95, 1.05)
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note.start *= stretch_factor
                note.end *= stretch_factor

        # 3. Note density alteration
        density_factor = random.uniform(0.95, 1.05)
        for instrument in midi_data.instruments:
            if density_factor < 1:
                notes_to_remove = int((1 - density_factor) * len(instrument.notes))
                for _ in range(notes_to_remove):
                    if instrument.notes:
                        instrument.notes.pop(random.randint(0, len(instrument.notes) - 1))
            else:
                notes_to_add = int((density_factor - 1) * len(instrument.notes))
                for _ in range(notes_to_add):
                    if instrument.notes:
                        semitone_offset = random.uniform(-2, 2)
                        reference_note = random.choice(instrument.notes)
                        new_note = pretty_midi.Note(
                            velocity=reference_note.velocity,
                            pitch=reference_note.pitch + int(semitone_offset),
                            start=reference_note.start - 0.1,
                            end=reference_note.start
                        )
                        instrument.notes.append(new_note)

        return midi_data
