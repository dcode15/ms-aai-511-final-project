from typing import Dict, List, Set

import numpy as np
import pretty_midi


class ChordFeatureExtractor:
    @staticmethod
    def extract_features(midi_data: pretty_midi.PrettyMIDI, time_step: float = 0.1) -> Dict[str, float]:
        end_time: float = midi_data.get_end_time()
        time_points: np.ndarray = np.arange(0, end_time, time_step)

        pitch_classes_over_time: List[Set[int]] = []
        chords_over_time: List[Set[int]] = []
        chord_durations: List[float] = []
        vertical_intervals: List[int] = []

        for t in time_points:
            active_notes: List[pretty_midi.Note] = [
                note for instrument in midi_data.instruments
                for note in instrument.notes
                if note.start <= t < note.end
            ]

            pitches: Set[int] = set(note.pitch for note in active_notes)
            pitch_classes: Set[int] = set(pitch % 12 for pitch in pitches)

            pitch_classes_over_time.append(pitch_classes)
            chords_over_time.append(pitches)

            pitch_list = sorted(list(pitches))
            for i in range(len(pitch_list)):
                for j in range(i + 1, len(pitch_list)):
                    vertical_intervals.append(pitch_list[j] - pitch_list[i])

        # Pitch classes
        avg_pitch_classes: float = np.mean([len(pc) for pc in pitch_classes_over_time])

        # Chord duration
        current_chord: Set[int] = set()
        current_duration: float = 0
        for i, chord in enumerate(chords_over_time):
            if chord != current_chord:
                if current_chord:
                    chord_durations.append(current_duration)
                current_chord = chord
                current_duration = time_step
            else:
                current_duration += time_step
        if current_duration > 0:
            chord_durations.append(current_duration)
        avg_chord_duration: float = np.mean(chord_durations) if chord_durations else 0

        # Intervals
        total_intervals: int = len(vertical_intervals)
        perfect_intervals: float = sum(
            i % 12 in {0, 5, 7} for i in vertical_intervals) / total_intervals if total_intervals else 0
        minor_seconds: float = sum(i % 12 == 1 for i in vertical_intervals) / total_intervals if total_intervals else 0
        thirds: float = sum(i % 12 in {3, 4} for i in vertical_intervals) / total_intervals if total_intervals else 0
        fifths: float = sum(i % 12 == 7 for i in vertical_intervals) / total_intervals if total_intervals else 0
        tritones: float = sum(i % 12 == 6 for i in vertical_intervals) / total_intervals if total_intervals else 0
        octaves: float = sum(i % 12 == 0 for i in vertical_intervals) / total_intervals if total_intervals else 0

        return {
            "avg_simultaneous_pitch_classes": avg_pitch_classes,
            "perfect_vertical_intervals": perfect_intervals,
            "vertical_minor_seconds": minor_seconds,
            "vertical_thirds": thirds,
            "vertical_fifths": fifths,
            "vertical_tritones": tritones,
            "vertical_octaves": octaves,
            "avg_chord_duration": avg_chord_duration
        }
