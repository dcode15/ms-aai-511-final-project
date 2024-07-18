import numpy as np
import pretty_midi


class TextureFeatureExtractor:

    @staticmethod
    def extract_features(midi_data: pretty_midi.PrettyMIDI, time_step: float = 0.1):
        """
        Extracts texture-related features from the given MIDI data.

        Args:
            midi_data (pretty_midi.PrettyMIDI): The MIDI data to analyze.
            time_step (float, optional): The time step for analysis. Defaults to 0.1 seconds.

        Returns:
            dict: A dictionary containing the extracted features:
                - max_independent_voices: The maximum number of unique pitches sounding simultaneously.
                - avg_independent_voices: The average number of unique pitches sounding simultaneously.
                - var_independent_voices: The standard deviation of the number of unique pitches sounding simultaneously.
                - avg_simultaneity: The average number of notes sounding simultaneously.
                - var_simultaneity: The standard deviation of the number of notes sounding simultaneously.
        """
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
