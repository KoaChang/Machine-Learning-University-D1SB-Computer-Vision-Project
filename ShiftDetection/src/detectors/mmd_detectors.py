from typing import Optional

from alibi_detect.cd import ContextMMDDrift
import torch

from .detector import ShiftDetectorWithFeatureExtractor, ConceptShiftDetectorWithFeatureExtractor
from .mmd_helpers import mmd, mmd_conditional, RBFKernel, KroneckerDeltaKernel


class MMDCovariateShiftDetector(ShiftDetectorWithFeatureExtractor):
    """A kernel maximum mean discrepancy (MMD) test.

    This test was proposed by

    [1] Gretton, A., et al. A kernel two-sample test. JMLR (2012).

    We currently do not expose the choice of kernel. It defaults to an RBF
    kernel with a lengthscale set via the median heuristic. A p-value can be
    bootstrapped via a permutation test.

    Args:
        feature_extractor: A pytorch model used as feature extractor.
        return_p_value: If `False`, the score method returns the raw value of
            the test statistic with the convention that higher values indicate
            a shift. If `True`, it returns one minus a p-value (to follow the
            same convention). Not all detectors support p-values.
        num_permutations: Number of permutations for permutation test.
        batch_size: Batch size used to iterate over datasets.
        num_preprocessing_workers: Number of workers used in data loaders.
        device: Device to use for computations inside the detector.
    """

    def __init__(self,
                 feature_extractor: Optional[torch.nn.Module] = None,
                 return_p_value: bool = False,
                 num_permutations: int = 1000,
                 batch_size: int = 32,
                 num_preprocessing_workers: int = 4,
                 device: str = "cpu") -> None:
        super().__init__(feature_extractor, return_p_value, batch_size,
                         num_preprocessing_workers, device)
        self._num_permutations = num_permutations if return_p_value else 0

    def _fit_with_features(self, X: torch.Tensor):
        self._X_ref = X

    def _score_with_features(self, X: torch.Tensor) -> float:
        score, p_val = mmd(self._X_ref, X, kernel=RBFKernel(),
                           num_permutations=self._num_permutations)
        if self._return_p_value:
            return 1. - p_val.item()
        else:
            return score.item()


class MMDConceptShiftDetector(ConceptShiftDetectorWithFeatureExtractor):

    def __init__(self,
                 prop_score_estimator: str = "SVM",
                 prop_score_cutoff: float = 0.0,
                 feature_extractor: Optional[torch.nn.Module] = None,
                 return_p_value: bool = False,
                 percentage_held: float = 0.25,
                 num_permutations: int = 1000,
                 batch_size: int = 32,
                 num_preprocessing_workers: int = 4,
                 device: str = "cpu") -> None:
        super().__init__(feature_extractor, return_p_value, batch_size,
                         num_preprocessing_workers, device)
        self._prop_score_estimator = prop_score_estimator
        self._prop_score_cutoff = prop_score_cutoff
        self._percentage_held = percentage_held
        self._num_permutations = num_permutations if return_p_value else 0

    def _fit_with_features(self, X: torch.Tensor, y: torch.Tensor):
        self._X_ref = X
        self._y_ref = y

    def _score_with_features(self, X: torch.Tensor, y: torch.Tensor) -> float:
        score, p_val = mmd_conditional(
            self._y_ref,
            self._X_ref,
            y,
            X,
            kernel_x=KroneckerDeltaKernel(),
            kernel_y=RBFKernel(),
            lam_0=1e-4,
            lam_1=1e-4,
            prop_score_estimator=self._prop_score_estimator,
            prop_score_cutoff=self._prop_score_cutoff,
            num_permutations=self._num_permutations,
            percentage_held=self._percentage_held
        )
        if self._return_p_value:
            return 1. - p_val.item()
        else:
            return score.item()


class AlibiConceptShiftDetector(ConceptShiftDetectorWithFeatureExtractor):

    def __init__(self,
                 feature_extractor: Optional[torch.nn.Module] = None,
                 return_p_value: bool = False,
                 percentage_held: float = 0.25,
                 num_permutations: int = 1000,
                 batch_size: int = 32,
                 num_preprocessing_workers: int = 4,
                 device: str = "cpu") -> None:
        super().__init__(feature_extractor, return_p_value, batch_size,
                         num_preprocessing_workers, device)
        self._percentage_held = percentage_held
        self._num_permutations = num_permutations if return_p_value else 0

    def _fit_with_features(self, X: torch.Tensor, y: torch.Tensor):
        self._X_ref = X
        self._y_ref = y
        self._alibi_detector = ContextMMDDrift(y.float().numpy(), X.numpy(), prop_c_held=self._percentage_held,
            n_permutations=self._num_permutations, backend="pytorch")

    def _score_with_features(self, X: torch.Tensor, y: torch.Tensor) -> float:
        p_val, score, _, _ = self._alibi_detector.score(y.float().numpy(), X.numpy())
        if self._return_p_value:
            return 1. - p_val
        else:
            return score