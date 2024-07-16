import os
import pickle
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np
import pretty_midi
import torch
from tqdm import tqdm


class MIDIFeatureExtractor:
    @staticmethod
    def estimate_chords(midi_data: pretty_midi.PrettyMIDI, min_notes: int = 3, time_step: float = 0.25) -> Dict[
        Tuple[int, ...], int]:
        chord_counter = defaultdict(int)
        end_time = midi_data.get_end_time()

        for time in np.arange(0, end_time, time_step):
            active_pitches = MIDIFeatureExtractor._get_active_pitches(midi_data, time)
            if len(active_pitches) >= min_notes:
                chord_counter[tuple(sorted(active_pitches))] += 1

        return dict(chord_counter)

    @staticmethod
    def _get_active_pitches(midi_data: pretty_midi.PrettyMIDI, time: float) -> Set[int]:
        return {note.pitch % 12 for instrument in midi_data.instruments
                if not instrument.is_drum
                for note in instrument.notes
                if note.start <= time < note.end}

    @staticmethod
    def calculate_polyphony(midi_data: pretty_midi.PrettyMIDI, step_size: float = 0.1) -> float:
        end_time = midi_data.get_end_time()
        times = np.arange(0, end_time, step_size)
        polyphony = [MIDIFeatureExtractor._count_notes_at_time(midi_data, time) for time in times]
        return float(np.mean(polyphony))

    @staticmethod
    def _count_notes_at_time(midi_data: pretty_midi.PrettyMIDI, time: float) -> int:
        return sum(1 for instrument in midi_data.instruments
                   for note in instrument.notes
                   if note.start <= time < note.end)

    @staticmethod
    def extract_features(midi_file: str, composer: str) -> Optional[Dict[str, Any]]:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
        except Exception as e:
            print(f"Error processing {midi_file}: {str(e)}")
            return None

        features = {'composer': composer}
        features.update(MIDIFeatureExtractor._extract_instrumentation_features(midi_data))
        features.update(MIDIFeatureExtractor._extract_rhythmic_features(midi_data))
        features.update(MIDIFeatureExtractor._extract_pitch_features(midi_data))
        features.update(MIDIFeatureExtractor._extract_harmony_features(midi_data))
        features['trend_matrix'] = midi_data.get_pitch_class_transition_matrix(normalize=True)
        features['polyphony'] = MIDIFeatureExtractor.calculate_polyphony(midi_data)
        features['piano_roll'] = MIDIFeatureExtractor._extract_piano_roll(midi_data)

        return features

    @staticmethod
    def _extract_instrumentation_features(midi_data: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        instruments = [inst.program for inst in midi_data.instruments if not inst.is_drum]
        return {
            'num_instruments': torch.tensor(len(set(instruments)), dtype=torch.float32),
            'percussion': torch.tensor(int(any(inst.is_drum for inst in midi_data.instruments)), dtype=torch.float32)
        }

    @staticmethod
    def _extract_rhythmic_features(midi_data: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        features = {}
        if midi_data.time_signature_changes:
            first_ts = midi_data.time_signature_changes[0]
            features['time_signature_numerator'] = torch.tensor(first_ts.numerator, dtype=torch.float32)
            features['time_signature_denominator'] = torch.tensor(first_ts.denominator, dtype=torch.float32)
        else:
            features['time_signature_numerator'] = torch.tensor(0, dtype=torch.float32)
            features['time_signature_denominator'] = torch.tensor(0, dtype=torch.float32)
        features['tempo'] = torch.tensor(midi_data.estimate_tempo(), dtype=torch.float32)

        note_durations = [note.end - note.start for inst in midi_data.instruments for note in inst.notes]
        features['avg_note_duration'] = torch.tensor(np.mean(note_durations), dtype=torch.float32)
        features['std_note_duration'] = torch.tensor(np.std(note_durations), dtype=torch.float32)

        return features

    @staticmethod
    def _extract_pitch_features(midi_data: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        all_notes = [note.pitch for inst in midi_data.instruments for note in inst.notes]

        return {
            'pitch_range': torch.tensor(max(all_notes) - min(all_notes), dtype=torch.float32),
            'avg_pitch': torch.tensor(np.mean(all_notes), dtype=torch.float32),
            'pitch_std': torch.tensor(np.std(all_notes), dtype=torch.float32),
            'pitch_class_hist': torch.tensor(midi_data.get_pitch_class_histogram(normalize=True), dtype=torch.float32)
        }

    @staticmethod
    def _extract_harmony_features(midi_data: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        chord_counter = MIDIFeatureExtractor.estimate_chords(midi_data)
        chord_types = defaultdict(int)
        for chord, count in chord_counter.items():
            chord_types[len(chord)] += count

        features = {
            'num_chords': torch.tensor(len(chord_counter), dtype=torch.float32),
            'chord_types': torch.tensor([chord_types.get(i, 0) for i in range(1, 13)], dtype=torch.float32),
        }

        if chord_types:
            features['most_common_chord_type'] = torch.tensor(max(chord_types, key=chord_types.get),
                                                              dtype=torch.float32)
        else:
            features['most_common_chord_type'] = torch.tensor(0, dtype=torch.float32)

        return features

    @staticmethod
    def _extract_piano_roll(midi_data: pretty_midi.PrettyMIDI, fs: int = 100) -> torch.Tensor:
        piano_roll = midi_data.get_piano_roll(fs=fs)
        return torch.tensor(piano_roll, dtype=torch.float32)

    @staticmethod
    def process_and_pickle(data_dir: str, composers: List[str]) -> List[Dict[str, Any]]:
        output_file = "../data/preprocessed_data.pkl"
        if os.path.exists(output_file):
            print(f"Loading existing processed data from {output_file}")
            with open(output_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Processing MIDI files...")
            all_features = []

            # Count total number of MIDI files
            total_files = sum(len([f for f in os.listdir(os.path.join(data_dir, composer))
                                   if f.endswith(('.mid', '.midi'))])
                              for composer in composers if os.path.isdir(os.path.join(data_dir, composer)))

            # Create progress bar
            with tqdm(total=total_files, desc="Processing MIDI files") as pbar:
                for composer in composers:
                    composer_dir = os.path.join(data_dir, composer)
                    if not os.path.isdir(composer_dir):
                        print(f"Directory not found for composer: {composer}")
                        continue

                    for midi_file in os.listdir(composer_dir):
                        if midi_file.endswith(('.mid', '.midi')):
                            file_path = os.path.join(composer_dir, midi_file)
                            features = MIDIFeatureExtractor.extract_features(file_path, composer)
                            if features is not None:
                                all_features.append(features)
                            pbar.update(1)

            print(f"Processing complete. Saving processed data to {output_file}")
            with open(output_file, 'wb') as f:
                pickle.dump(all_features, f)

            return all_features


if __name__ == "__main__":
    data_directory = "../data/Composer_Dataset_2/"
    composers = ["Bach", "Beethoven", "Chopin", "Mozart"]
    output_pickle = "midi_features.pkl"

    features = MIDIFeatureExtractor.process_and_pickle(data_directory, composers)
    print(f"Features for {len(features)} MIDI files loaded/processed.")
