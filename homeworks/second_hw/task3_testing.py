import pytest
from PIL import Image
from homeworks.second_hw.task3 import *


@pytest.fixture
def sample_image():
    return Image.new('RGB', (300, 300), color='white')


def test_random_crop(sample_image):
    transform = RandomCrop(p=1.0, crop_size=(100, 100))
    cropped_image = transform(sample_image)
    assert cropped_image.size == (100, 100), "Crop size is incorrect"


def test_random_rotate(sample_image):
    transform = RandomRotate(p=1.0, degrees=45)
    rotated_image = transform(sample_image)
    assert rotated_image.size == sample_image.size, "Image size should remain the same after rotation"


def test_random_zoom(sample_image):
    transform = RandomZoom(p=1.0, zoom_range=(0.5, 2.0))
    zoomed_image = transform(sample_image)
    assert zoomed_image.size == sample_image.size, "Image size should remain the same after zoom"


def test_compose_transforms(sample_image):
    transform_pipeline = Compose([
        RandomCrop(p=1.0, crop_size=(100, 100)),
        RandomRotate(p=1.0, degrees=30)
    ])
    transformed_image = transform_pipeline(sample_image)
    assert transformed_image.size == (100, 100), "Compose pipeline did not apply transformations correctly"


def test_to_tensor(sample_image):
    transform = ToTensor()
    tensor_image = transform(sample_image)
    assert isinstance(tensor_image, torch.Tensor), "Output is not a tensor"
    assert tensor_image.shape == (3, 300, 300), "Tensor shape is incorrect"
