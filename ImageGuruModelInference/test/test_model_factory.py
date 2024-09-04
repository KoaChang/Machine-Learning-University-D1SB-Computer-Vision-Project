import os
import tempfile
import torch
from torchvision.models.resnet import ResNet, Bottleneck
import unittest

from image_guru_model_inference.model_factory import create_model
from image_guru_model_inference.models.pytorch_predictor import PyTorchPredictor


class TestCreateModel(unittest.TestCase):

    def test_PyTorchPredictor(self):
        classes = ['class-1', 'class-2', 'class-3']

        # create a Resnet50 model with randomly initialized weights
        reset50 = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=len(classes))
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as fd:
            torch.save(reset50.state_dict(), fd)
            model_path = fd.name

        model_context = {
            'predictor_class': 'PyTorchPredictor',
            'model_path': model_path,
            'classes': classes,
            'model_type': 'resnet50',
            'multi_label': False
        }

        model = create_model(model_context)

        self.assertTrue(isinstance(model, PyTorchPredictor))

        os.unlink(model_path)
