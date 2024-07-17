import numpy as np
import pretty_midi


class TextureFeatureExtractor:
    @staticmethod
    def extract_features(midi_data: pretty_midi.PrettyMIDI, time_step: float = 0.1):
        notes = [note for instrument in midi_data.instruments for note in instrument.notes]
        notes.sort(key=lambda x: x.start)

        end_time = max(note.end for note in notes)
        time_points = np.arange(0, end_time, time_step)

        voices = np.zeros(len(time_points))
        simultaneity = np.zeros(len(time_points))

        for i, t in enumerate(time_points):
            active_notes = [note for note in notes if note.start <= t < note.end]
            voices[i] = len(set(note.pitch for note in active_notes))
            simultaneity[i] = len(active_notes)

        # Voices
        max_voices = np.max(voices)
        avg_voices = np.mean(voices[voices > 0])
        var_voices = np.std(voices[voices > 0])

        # Simultaneity
        avg_simultaneity = np.mean(simultaneity)
        var_simultaneity = np.std(simultaneity)

        return {
            "max_independent_voices": max_voices,
            "avg_independent_voices": avg_voices,
            "var_independent_voices": var_voices,
            "avg_simultaneity": avg_simultaneity,
            "var_simultaneity": var_simultaneity
        }
