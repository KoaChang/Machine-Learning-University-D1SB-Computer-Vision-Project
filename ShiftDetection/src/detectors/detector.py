import torch
from torch.utils.data import Dataset, DataLoader

from typing import Optional, Tuple


def collate_inputs_only(batch):
    """collate_fn that only batches the inputs x."""
    return (torch.stack([item[0] for item in batch], dim=0),)


class ShiftDetector(object):
    """Base class for shift detectors.

    Args:
        return_p_value: If `False`, the score method returns the raw value of
            the test statistic with the convention that higher values indicate
            a shift. If `True`, it returns one minus a p-value (to follow the
            same convention). Not all detectors support p-values.
        batch_size: Batch size used to iterate over datasets.
        num_preprocessing_workers: Number of workers used in data loaders.
        device: Device to use for computations inside the detector.
    """

    def __init__(self,
                 return_p_value: bool = False,
                 batch_size: int = 32,
                 num_preprocessing_workers: int = 4,
                 device: str = "cpu") -> None:
        self._return_p_value = return_p_value
        self.batch_size = batch_size
        self.num_preprocessing_workers = num_preprocessing_workers
        self.device = device

    def fit(self, dataset: Dataset) -> None:
        """Fit the detector to a reference dataset."""
        raise NotImplementedError()

    def score(self, dataset: Dataset) -> float:
        """Compute distribution shift score for a query dataset."""
        raise NotImplementedError()

    def make_data_loader(self,
                         dataset: Dataset,
                         shuffle: bool = False,
                         labels: bool = False) -> DataLoader:
        """Return a data loader to iterate over a dataset.

        Args:
            dataset: The dataset.
            shuffle: Whether to shuffle or not.
            labels: If True, the dataloader will return labels.
        """
        collate_fn = None if labels else collate_inputs_only
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_preprocessing_workers,
                          collate_fn=collate_fn)


class ShiftDetectorWithFeatureExtractor(ShiftDetector):
    """Base class for detectors working on extracted features.

    These shift detectors extract some (lower-dimensional) features from
    the datasets, which are then used as inputs to the shift detection
    methods. Subclasses have to overwrite `fit_with_features` and
    `score_with_features`.

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
        super(ShiftDetectorWithFeatureExtractor, self).__init__(
            return_p_value, batch_size, num_preprocessing_workers, device)
        self.feature_extractor = feature_extractor or torch.nn.Identity()
        self.feature_extractor = self.feature_extractor.to(self.device)

    def fit(self, dataset: Dataset) -> None:
        """Fit the detector to a reference dataset."""
        X = self.extract_features(dataset)
        self._fit_with_features(X)

    def score(self, dataset: Dataset) -> float:
        """Compute distribution shift score for a query dataset."""
        X = self.extract_features(dataset)
        return self._score_with_features(X)

    @torch.no_grad()
    def extract_features(self, dataset: Dataset) -> torch.Tensor:
        """Extract features from a dataset."""
        dataloader = self.make_data_loader(dataset, labels=False)
        Xs = []
        for batch in dataloader:
            inputs = batch[0].to(self.device)
            Xs.append(self.feature_extractor(inputs))
        X = torch.cat(Xs, dim=0).cpu()
        return X

    def _fit_with_features(self, X: torch.Tensor) -> None:
        raise NotImplementedError()

    def _score_with_features(self, X: torch.Tensor) -> float:
        raise NotImplementedError()


class ConceptShiftDetectorWithFeatureExtractor(ShiftDetector):
    """Base class for concept shift detectors working on extracted features.

    These shift detectors extract some (lower-dimensional) features from
    the datasets, which are then used as inputs to the shift detection
    methods. Subclasses have to overwrite `fit_with_features` and
    `score_with_features`.

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
        super(ConceptShiftDetectorWithFeatureExtractor, self).__init__(
            return_p_value, batch_size, num_preprocessing_workers, device)
        self.feature_extractor = feature_extractor or torch.nn.Identity()
        self.feature_extractor = self.feature_extractor.to(self.device)

    def fit(self, dataset: Dataset) -> None:
        """Fit the detector to a reference dataset."""
        X, y = self.extract_features_and_labels(dataset)
        self._fit_with_features(X, y)

    def score(self, dataset: Dataset) -> float:
        """Compute distribution shift score for a query dataset."""
        X, y = self.extract_features_and_labels(dataset)
        return self._score_with_features(X, y)

    @torch.no_grad()
    def extract_features_and_labels(self, dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from a dataset."""
        dataloader = self.make_data_loader(dataset, labels=True)
        Xs = []
        ys = []
        for batch in dataloader:
            inputs = batch[0].to(self.device)
            Xs.append(self.feature_extractor(inputs))
            ys.append(batch[1])
        X = torch.cat(Xs, dim=0).cpu()
        y = torch.cat(ys, dim=0).cpu()
        if y.dim() == 1:
            y = y.unsqueeze(1)
        return X, y

    def _fit_with_features(self, X: torch.Tensor, y: torch.Tensor) -> None:
        raise NotImplementedError()

    def _score_with_features(self, X: torch.Tensor, y: torch.Tensor) -> float:
        raise NotImplementedError()
