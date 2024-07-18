import os
import pickle
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm

from chord_feature_extractor import ChordFeatureExtractor
from melodic_feature_extractor import MelodicFeatureExtractor
from multidimensional_feature_extractor import MultidimensionalFeatureExtractor
from pitch_feature_extractor import PitchFeatureExtractor
from rhythm_feature_extractor import RhythmFeatureExtractor
from texture_feature_extractor import TextureFeatureExtractor


class FeatureExtractor:
    @staticmethod
    def extract_features(midi_file: str, time_step: float = 0.1) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        features: Dict[str, Any] = {}
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        features.update(TextureFeatureExtractor.extract_features(midi_data, time_step))
        features.update(RhythmFeatureExtractor.extract_features(midi_data))
        features.update(PitchFeatureExtractor.extract_features(midi_data))
        features.update(MelodicFeatureExtractor.extract_features(midi_data))
        features.update(ChordFeatureExtractor.extract_features(midi_data, time_step))

        return features, MultidimensionalFeatureExtractor.extract_features(midi_data)

    @staticmethod
    def extract_features_for_multiple_files(
            data_directory: str,
            composers: List[str],
            sampling_frequency: int = 10,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        time_step: float = 1 / sampling_frequency
        output_file = os.path.join(data_directory, "extracted_features.pkl")

        if os.path.exists(output_file):
            print(f"Loading existing features from {output_file}")
            with open(output_file, 'rb') as f:
                return pickle.load(f)

        all_scalar_features = []
        all_multidimensional_features = []
        midi_files = FeatureExtractor._get_midi_files(data_directory, composers)

        with tqdm(total=len(midi_files), desc="Processing MIDI files") as pbar:
            for midi_file, composer in midi_files:
                pbar.set_postfix_str(f"Processing: {os.path.basename(midi_file)}")

                scalar_features, multidimensional_features = FeatureExtractor.extract_features(midi_file, time_step)
                scalar_features['file_name'] = os.path.basename(midi_file)
                scalar_features['composer'] = composer
                all_scalar_features.append(scalar_features)
                all_multidimensional_features.append(multidimensional_features)
                if len(all_scalar_features) >= 30:
                    break

                pbar.update(1)

        df = pd.DataFrame(all_scalar_features)
        with open(output_file, 'wb') as f:
            pickle.dump(df, f)
        print(f"Saved extracted features to {output_file}")

        return df, all_multidimensional_features

    @staticmethod
    def _get_midi_files(data_directory: str, composers: List[str]) -> List[tuple[str, str]]:
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
