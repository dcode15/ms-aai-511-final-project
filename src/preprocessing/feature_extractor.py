import os
import pickle
import traceback
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm

from src.preprocessing.chord_feature_extractor import ChordFeatureExtractor
from src.preprocessing.melodic_feature_extractor import MelodicFeatureExtractor
from src.preprocessing.multidimensional_feature_extractor import MultidimensionalFeatureExtractor
from src.preprocessing.pitch_feature_extractor import PitchFeatureExtractor
from src.preprocessing.rhythm_feature_extractor import RhythmFeatureExtractor
from src.preprocessing.texture_feature_extractor import TextureFeatureExtractor


class FeatureExtractor:

    @staticmethod
    def extract_features(midi_data: pretty_midi.PrettyMIDI, time_step: float = 0.1) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Extracts scalar and multidimensional features from a single MIDI file.

        Args:
            midi_file (str): Path to the MIDI file.
            time_step (float, optional): Time step for feature extraction. Defaults to 0.1 seconds.

        Returns:
            Tuple[Dict[str, Any], Dict[str, np.ndarray]]: A tuple containing:
                - A dictionary of scalar features.
                - A dictionary of multidimensional features.
        """
        features: Dict[str, Any] = {}

        features.update(TextureFeatureExtractor.extract_features(midi_data, time_step))
        features.update(RhythmFeatureExtractor.extract_features(midi_data))
        features.update(PitchFeatureExtractor.extract_features(midi_data))
        features.update(MelodicFeatureExtractor.extract_features(midi_data))
        features.update(ChordFeatureExtractor.extract_features(midi_data, time_step))
        features["length"] = midi_data.get_end_time()

        return features, MultidimensionalFeatureExtractor.extract_features(midi_data)

    @staticmethod
    def extract_features_for_directory(
            data_directory: str,
            composers: List[str],
            sampling_frequency: int = 10,
    ) -> Tuple[pd.DataFrame, List[Dict[str, np.ndarray]]]:
        """
        Extracts features from multiple MIDI files for specified composers. This method caches the extracted features to
        a file named "extracted_features.pkl" in the data directory. If this file exists, it will load the features from
        the file instead of reprocessing the MIDI files.

        Args:
            data_directory (str): Root directory containing composer subdirectories with MIDI files.
            composers (List[str]): List of composer names to process.
            sampling_frequency (int, optional): Sampling frequency for feature extraction. Defaults to 10 Hz.

        Returns:
            Tuple[pd.DataFrame, List[Dict[str, np.ndarray]]]: A tuple containing:
                - A DataFrame of scalar features for all processed MIDI files.
                - A list of dictionaries containing multidimensional features for all processed MIDI files.
        """
        time_step: float = 1 / sampling_frequency
        output_file = os.path.join(data_directory, "extracted_features.pkl")

        if os.path.exists(output_file):
            print(f"Loading existing features from {output_file}")
            with open(output_file, 'rb') as processed_data_file:
                return pickle.load(processed_data_file)

        all_scalar_features = []
        all_multidimensional_features = []
        midi_files = FeatureExtractor._get_midi_files(data_directory, composers)

        with tqdm(total=len(midi_files), desc="Processing MIDI files") as pbar:
            for midi_file, composer in midi_files:
                pbar.set_postfix_str(f"Processing: {os.path.basename(midi_file)}")

                try:
                    midi_data = pretty_midi.PrettyMIDI(midi_file)
                    scalar_features, multidimensional_features = FeatureExtractor.extract_features(midi_data, time_step)
                    scalar_features['file_name'] = os.path.basename(midi_file)
                    scalar_features['composer'] = composer
                    all_scalar_features.append(scalar_features)
                    all_multidimensional_features.append(multidimensional_features)
                except Exception as e:
                    print(f"Failed to process file {os.path.basename(midi_file)}.")
                    print(f"Error message: {str(e)}")
                    print("Stack trace:")
                    traceback.print_exc()

                pbar.update(1)

        scalar_df = pd.DataFrame(all_scalar_features)
        with open(output_file, 'wb') as processed_data_file:
            pickle.dump((scalar_df, all_multidimensional_features), processed_data_file)
        print(f"Saved extracted features to {output_file}")

        return scalar_df, all_multidimensional_features

    @staticmethod
    def _get_midi_files(data_directory: str, composers: List[str]) -> List[Tuple[str, str]]:
        """
        Retrieves all MIDI files for the specified composers from the given directory. The method searches for files
        with '.mid' or '.midi' extensions in the specified composer directories and their subdirectories.

        Args:
            data_directory (str): Root directory containing composer subdirectories.
            composers (List[str]): List of composer names to process.

        Returns:
            List[Tuple[str, str]]: A list of tuples, each containing:
                - The full path to a MIDI file.
                - The name of the composer.
        """
        midi_files = []
        for composer in composers:
            composer_dir = os.path.join(data_directory, composer)
            if not os.path.isdir(composer_dir):
                print(f"Warning: Directory not found for composer {composer}")
                continue

            for root, _, files in os.walk(composer_dir):
                for file in files:
                    if file.lower().endswith(('.mid', '.midi')):
                        midi_files.append((os.path.join(root, file), composer))
        return midi_files
