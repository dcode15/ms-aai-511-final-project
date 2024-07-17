import os
from typing import Dict, Any, List

import pandas as pd
import pretty_midi
from tqdm import tqdm

from chord_feature_extractor import ChordFeatureExtractor
from melodic_feature_extractor import MelodicFeatureExtractor
from pitch_feature_extractor import PitchFeatureExtractor
from rhythm_feature_extractor import RhythmFeatureExtractor
from texture_feature_extractor import TextureFeatureExtractor


class FeatureExtractor:
    @staticmethod
    def extract_features(midi_file: str, time_step: float = 0.1) -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        features.update(TextureFeatureExtractor.extract_features(midi_data, time_step))
        features.update(RhythmFeatureExtractor.extract_features(midi_data))
        features.update(PitchFeatureExtractor.extract_features(midi_data))
        features.update(MelodicFeatureExtractor.extract_features(midi_data))
        features.update(ChordFeatureExtractor.extract_features(midi_data, time_step))

        return features

    @staticmethod
    def extract_features_for_multiple_files(
            data_directory: str,
            composers: List[str],
            time_step: float = 0.1
    ) -> pd.DataFrame:
        output_file = os.path.join(data_directory, "extracted_features.csv")

        if os.path.exists(output_file):
            print(f"Loading existing features from {output_file}")
            return pd.read_csv(output_file)

        all_features = []
        midi_files = FeatureExtractor._get_midi_files(data_directory, composers)

        with tqdm(total=len(midi_files), desc="Processing MIDI files") as pbar:
            for midi_file, composer in midi_files:
                pbar.set_postfix_str(f"Processing: {os.path.basename(midi_file)}")

                features = FeatureExtractor._process_file(midi_file, composer, time_step)
                all_features.append(features)
                if len(all_features) >= 30:
                    break

                pbar.update(1)

        df = pd.DataFrame(all_features)
        df.to_csv(output_file, index=False)
        print(f"Saved extracted features to {output_file}")

        return df

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

    @staticmethod
    def _process_file(
            midi_file: str,
            composer: str,
            time_step: float
    ) -> Dict[str, Any]:
        features = FeatureExtractor.extract_features(midi_file, time_step)
        features['file_name'] = os.path.basename(midi_file)
        features['composer'] = composer
        return features
