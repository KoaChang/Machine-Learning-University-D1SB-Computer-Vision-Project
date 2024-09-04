import torch
import torchdrift

from typing import Optional

from .detector import ShiftDetectorWithFeatureExtractor


class KSCovariateShiftDetector(ShiftDetectorWithFeatureExtractor):
    """A detector based on elementwise Komolgorov-Smirnov tests.

    This is a wrapper around the detector implemented in torchdrift, see
    https://github.com/TorchDrift/TorchDrift. It applies a univariate KS test
    to the marginal distribution of each feature and returns the maximum score
    across all features, as proposed by S. Rabanser et al: "Failing Loudly: An
    Empirical Study of Methods for Detecting Dataset Shift" (NeurIPS), 2019.

    Args:
        feature_extractor: A pytorch model used as feature extractor.
        return_p_value: If `False`, the score method returns the raw value of
            the test statistic with the convention that higher values indicate
            a shift. If `True`, it returns one minus a p-value (to follow the
            same convention). Not all detectors support p-values.
        batch_size: Batch size used to iterate over datasets.
        num_preprocessing_workers: Number of workers used in data loaders.
        device: Device to use for computations inside the detector.
    """

    def __init__(self,
                 feature_extractor: Optional[torch.nn.Module] = None,
                 return_p_value: bool = False,
                 batch_size: int = 32,
                 num_preprocessing_workers: int = 4,
                 device: str = "cpu") -> None:
        super(KSCovariateShiftDetector, self).__init__(
            feature_extractor, return_p_value, batch_size,
            num_preprocessing_workers, device)
        self.torchdrift_detector = torchdrift.detectors.KSDriftDetector(
            return_p_value=return_p_value)

    def _fit_with_features(self, X: torch.Tensor):
        self.torchdrift_detector.fit(X)

    def _score_with_features(self, X: torch.Tensor):
        if self._return_p_value:
            return 1. - self.torchdrift_detector(X)
        else:
            return self.torchdrift_detector(X).item()
