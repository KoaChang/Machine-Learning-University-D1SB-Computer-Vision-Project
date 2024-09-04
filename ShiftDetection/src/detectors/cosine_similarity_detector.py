import torch

from typing import Optional

from .detector import ShiftDetectorWithFeatureExtractor


class CosineSimilarityCovariateShiftDetector(
        ShiftDetectorWithFeatureExtractor):
    """A shift detection heuristic based on cosine similarity of features.

    The OOD score of this detector is one minus the average pairwise cosine
    similarity between the features of points in query and reference set.
    This has been proposed in an AWS blogpost:
    https://aws.amazon.com/blogs/machine-learning/detect-nlp-data-drift-using-custom-amazon-sagemaker-model-monitor/
    It does not support the computation of a p-value.

    Args:
        feature_extractor: A pytorch model used as feature extractor.
        return_p_value: Present for compatibility. This detector does not
            support p-values and setting `True` will raise an exception.
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
        if return_p_value:
            raise ValueError("This detector does not support p-values.")
        super(CosineSimilarityCovariateShiftDetector, self).__init__(
            feature_extractor, return_p_value, batch_size,
            num_preprocessing_workers, device)

    def _fit_with_features(self, X: torch.Tensor) -> None:
        self._X_ref = X
        self._x_ref_norms = X.pow(2).sum(dim=1).sqrt()

    def _score_with_features(self, X: torch.Tensor) -> float:
        cosine_similarities = X @ self._X_ref.T
        cosine_similarities /= X.pow(2).sum(dim=1).sqrt().view(-1, 1)
        cosine_similarities /= self._x_ref_norms.view(1, -1)
        return 0.5 * torch.mean(1. - cosine_similarities).item()
