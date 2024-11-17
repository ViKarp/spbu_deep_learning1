import random
import torch
from PIL import Image
import numpy as np


class BaseTransform:
    def __init__(self, p: float):
        """
        Base class for all transformations.

        :param p: Probability of applying the transformation.
        """
        assert 0.0 <= p <= 1.0, "Probability p must be between 0 and 1."
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError("BaseTransform is an abstract class.")


class RandomCrop(BaseTransform):
    def __init__(self, p: float, crop_size: tuple[int, int]):
        """
        Randomly crops the image.

        :param p: Probability of applying the transformation.
        :param crop_size: Size of the crop (width, height).
        """
        super().__init__(p)
        self.crop_size = crop_size

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            width, height = image.size
            crop_width, crop_height = self.crop_size
            if crop_width > width or crop_height > height:
                return image
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            right = left + crop_width
            bottom = top + crop_height
            return image.crop((left, top, right, bottom))
        return image


class RandomRotate(BaseTransform):
    def __init__(self, p: float, degrees: float):
        """
        Randomly rotates the image.

        :param p: Probability of applying the transformation.
        :param degrees: Maximum rotation angle in degrees.
        """
        super().__init__(p)
        self.degrees = degrees

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            return image.rotate(angle)
        return image


class RandomZoom(BaseTransform):
    def __init__(self, p: float, zoom_range: tuple[float, float]):
        """
        Randomly zooms the image.

        :param p: Probability of applying the transformation.
        :param zoom_range: Zoom range as a tuple (min_zoom, max_zoom).
        """
        super().__init__(p)
        assert zoom_range[0] > 0 and zoom_range[1] > 0, "Zoom range values must be greater than 0."
        self.zoom_range = zoom_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            width, height = image.size
            zoom_factor = random.uniform(*self.zoom_range)
            new_width = int(width * zoom_factor)
            new_height = int(height * zoom_factor)
            zoomed_image = image.resize((new_width, new_height), Image.LANCZOS)

            left = (new_width - width) // 2
            top = (new_height - height) // 2
            right = left + width
            bottom = top + height

            return zoomed_image.crop((left, top, right, bottom))
        return image


class ToTensor(BaseTransform):
    def __init__(self):
        """
        Converts a PIL Image to a torch.Tensor.
        """
        super().__init__(1.0)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image_np = np.array(image).astype(np.float32) / 255.0
        #image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        return image_np


class Compose:
    def __init__(self, transforms: list[BaseTransform]):
        """
        Composes a list of transformations.

        :param transforms: List of transformations to apply.
        """
        self.transforms = transforms

    def __call__(self, image: Image.Image) -> Image.Image:
        print(image.size)
        for transform in self.transforms:
            image = transform(image)
            print(image.size)
        return image


image_path = "images.jpg"
image = Image.open(image_path)

transform_pipeline = Compose([
    RandomCrop(p=0.8, crop_size=(200, 200)),
    RandomRotate(p=0.5, degrees=30),
    RandomZoom(p=0.7, zoom_range=(0.8, 1.2)),
    ToTensor()
])

transformed_image = transform_pipeline(image)

print(transformed_image)
