from collections import Counter
from typing import Dict, List

import numpy as np
import pretty_midi


class MelodicFeatureExtractor:
    @staticmethod
    def extract_features(midi_data: pretty_midi.PrettyMIDI) -> Dict[str, float]:
        notes: List[pretty_midi.Note] = sorted(
            [note for instrument in midi_data.instruments for note in instrument.notes],
            key=lambda x: x.start
        )

        intervals: List[int] = []
        for i in range(1, len(notes)):
            intervals.append(notes[i].pitch - notes[i - 1].pitch)

        if not intervals:
            return {
                "average_melodic_interval": 0,
                "most_common_melodic_interval": 0,
                "amount_of_arpeggiation": 0,
                "stepwise_motion": 0,
                "direction_of_motion": 0,
            }

        # Average Melodic Interval
        avg_melodic_interval: float = np.mean(np.abs(intervals))

        # Most Common Melodic Interval
        most_common_interval: int = int(Counter(np.abs(intervals)).most_common(1)[0][0])

        # Amount of Arpeggiation
        arpeggio_intervals = {0, 3, 4, 7, 10, 11, 12, 15, 16}
        arpeggiation: float = sum(abs(i) in arpeggio_intervals for i in intervals) / len(intervals)

        # Stepwise Motion
        stepwise: float = sum(abs(i) in {1, 2} for i in intervals) / len(intervals)

        # Direction of Motion
        rising: float = sum(i > 0 for i in intervals) / len(intervals)

        return {
            "average_melodic_interval": avg_melodic_interval,
            "most_common_melodic_interval": most_common_interval,
            "amount_of_arpeggiation": arpeggiation,
            "stepwise_motion": stepwise,
            "direction_of_motion": rising,
        }
