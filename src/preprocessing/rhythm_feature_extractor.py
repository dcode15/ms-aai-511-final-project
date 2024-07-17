import numpy as np
import pretty_midi


class RhythmFeatureExtractor:
    @staticmethod
    def extract_features(midi_data: pretty_midi.PrettyMIDI):
        notes = [note for instrument in midi_data.instruments for note in instrument.notes]

        # Note Density
        total_duration = max(note.end for note in notes) - min(note.start for note in notes)
        note_density = len(notes) / total_duration

        # Note Duration
        durations = np.array([note.end - note.start for note in notes])
        avg_note_duration = np.mean(durations)
        var_note_duration = np.std(durations)

        # Initial Tempo
        tempo = midi_data.estimate_tempo()

        # Time Signature
        time_signature_changes = midi_data.time_signature_changes
        time_signature_numerator = None
        time_signature_denominator = None
        if time_signature_changes:
            time_signature_numerator = time_signature_changes[0].numerator
            time_signature_denominator = time_signature_changes[0].denominator

        return {
            "note_density": note_density,
            "avg_note_duration": avg_note_duration,
            "var_note_duration": var_note_duration,
            "initial_tempo": tempo,
            "time_signature_numerator": time_signature_numerator,
            "time_signature_denominator": time_signature_denominator,
        }
