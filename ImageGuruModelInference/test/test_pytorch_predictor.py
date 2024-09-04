import numpy as np
import os
from PIL import Image
import tempfile
import torch
from torchvision.models.resnet import ResNet, Bottleneck
import unittest

from image_guru_model_inference.models.pytorch_predictor import PyTorchPredictor


class TestPyTorchPredictor(unittest.TestCase):

    def _create_random_image(self):
        np.random.seed(0)
        imarr = 255 * np.random.rand(512, 512, 3)
        imarr = imarr.astype('uint8')
        img = Image.fromarray(imarr).convert("RGB")
        return img

    def test_predict_multi_class(self):
        # create a dummy image
        img = self._create_random_image()

        # create a Resnet50 model with randomly initialized weights
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=3)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as fd:
            torch.save(model.state_dict(), fd)
            model_path = fd.name

        classes = ['class-1', 'class-2', 'class-3']

        # initialize and call predictor
        predictor = PyTorchPredictor(model_path, classes, model_type='resnet50', device='cpu',
                                     transforms='default', multi_label=False)
        result = predictor.predict(samples=[img], sample_type='pil')
        output_classes, output_probabilities, metadata = result
        os.unlink(model_path)

        # verify results
        self.assertEqual(len(output_classes), 1)
        self.assertEqual(len(output_classes[0]), 1)
        self.assertIn(output_classes[0][0], classes)
        self.assertEqual(len(output_probabilities), 1)
        self.assertEqual(len(output_probabilities[0]), 1)
        self.assertGreaterEqual(output_probabilities[0][0], 0)
        self.assertLessEqual(output_probabilities[0][0], 1)

    def test_predict_multi_label(self):
        # create a dummy image
        img = self._create_random_image()

        # create a Resnet50 model with randomly initialized weights
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=3)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as fd:
            torch.save(model.state_dict(), fd)
            model_path = fd.name

        classes = ['class-1', 'class-2', 'class-3']

        # initialize and call predictor
        predictor = PyTorchPredictor(model_path, classes, model_type='resnet50', device='cpu',
                                     transforms='default', multi_label=True)
        result = predictor.predict(samples=[img], sample_type='pil')
        output_classes, output_probabilities, metadata = result
        os.unlink(model_path)

        # verify results
        self.assertEqual(len(output_classes), 1)
        self.assertIn(len(output_classes[0]), [0, 1, 2, 3])
        for c in output_classes[0]:
            self.assertIn(c, classes)
        self.assertEqual(len(output_probabilities), 1)
        self.assertIn(len(output_probabilities[0]), [0, 1, 2, 3])
        self.assertEqual(len(output_probabilities[0]), len(output_classes[0]))
        for p in output_probabilities[0]:
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 1)
