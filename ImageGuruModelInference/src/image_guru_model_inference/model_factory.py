from image_guru_model_inference.models.pytorch_predictor import PyTorchPredictor


def create_model(model_context, device=None):
    """
    A 'factory' method to create the predictor instance for a given product type.

    :param model_context: A dict containing the model information as key-value pairs.
                          The following keys are expected:
                          1. model_path: Path to the model file
                          2. classes: A list of class names. The order is important and should correspond to the predictor output.
                          3. model_type: Type of model. Should be one of the supported model types by the predictor.
                          4. multi_label: True if the model is a multi-label model, False for a multi-class (single label output) model.
                          5. predictor_class: The predictor class to use, e.g. 'PyTorchPredictor'
                          This dict could potentially be compiled by the caller by data stored in S3, DynammoDB, etc.
    :param device: GPU ID (int) if a GPU is to used, 'cpu' otherwise. If None, the predictor will decide dynamically
                   based on the GPU availability.
    """
    try:
        predictor_class = model_context['predictor_class']
    except KeyError:
        raise RuntimeError("The model_context must have a 'predictor_class' key containing the predictor classes "
                           "such as 'PyTorchPredictor'")

    if predictor_class == 'PyTorchPredictor':
        model = PyTorchPredictor(model_context['model_path'], model_context['classes'],
                                 model_type=model_context['model_type'], device=device, transforms='default',
                                 multi_label=model_context['multi_label'])
    else:
        raise RuntimeError('Unsupported predictor_class: {}'.format(predictor_class))

    return model
