import random
from typing import Union, Optional, Iterable, List

from ddsketch import DDSketch
from tdigest import TDigest

def add_to_tdigest(salience: List[float], tdigest: TDigest) -> None:
    if len(salience) > 0:
        if len(salience) == 1:
            tdigest.update(float(salience[0]))
            if random.random() < 0.01:
                tdigest.compress()  # Perform tdigest maintenance once in a while
        else:
            tdigest.batch_update(salience)  # this always does maintenance


class PercentileEstimator:
    """
    Wraps tdigest and ddsketch for migration to ddsketch from serlialized tdigests.
    """
    def __init__(self, est: Optional[Union[TDigest, DDSketch]] = None):
        if est is None:
            est = DDSketch()
        self._est = est
        self.is_ddsketch = isinstance(est, DDSketch)

    def percentile(self, q: float) -> Optional[float]:
        """Get percentile value for q in [0, 100]"""
        if self.is_ddsketch:
            return self._est.get_quantile_value(q / 100)
        else:
            sketch: TDigest = self._est
            return sketch.percentile(q)

    def add(self, vals: List[float]) -> None:
        """Add value to sketch"""
        if self.is_ddsketch:
            for v in vals:
                self._est.add(v)
        else:
            sketch: TDigest = self._est
            add_to_tdigest(vals, sketch)


    @property
    def count(self) -> int:
        """Get number of values in sketch"""
        if self.is_ddsketch:
            return int(self._est.count)
        else:
            sketch: TDigest = self._est
            return sketch.n
