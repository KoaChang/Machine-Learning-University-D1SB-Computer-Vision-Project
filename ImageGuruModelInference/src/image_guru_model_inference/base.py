from abc import ABC, abstractmethod


class PredictorBase(ABC):
    """
    Base class for all the predictors.
    Not expected to be used directly. Clients should instead use one of the derived classes.
    """

    @abstractmethod
    def predict(self, samples):
        """
        Run prediction on the provided samples. Derived classes must implement this method.

        :param samples: List of samples. The type of each element is set by the derived class.
        :type samples: list
        :return: A tuple of (output_classes, output_probabilities, metadata).
        output_classes is a list of lists.
        Outer list is over the samples.
        Inner list, corresponding to a sample, contains a list of predicted classes.
        output_probabilities is a list of lists.
        Outer list is over the samples.
        Inner list, corresponding to a sample, contains a list of probabilities corresponding to the predicted classes.
        metadata is a dict containing any metadata.
        :rtype: tuple
        """
        pass
