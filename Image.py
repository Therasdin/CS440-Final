from PIL import Image as PILImage
import numpy as np
from Pixel import Pixel

class Image:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.width = 0
        self.height = 0
        self.pixels = []

        if filepath:
            self.load(filepath)

    def load(self, filepath):
        """Load a .jpg image and store pixel data."""
        img = PILImage.open(filepath).convert("RGB")
        self.width, self.height = img.size
        img_data = np.array(img)

        # Create a 2D list of Pixel objects
        self.pixels = [
            [Pixel(r, g, b) for (r, g, b) in row]
            for row in img_data
        ]

    def to_numpy(self, normalize=False):
        """Convert pixel data to a NumPy array (for ML models)."""
        arr = np.array(
            [[[p.r, p.g, p.b] for p in row] for row in self.pixels],
            dtype=np.float32 if normalize else np.uint8
        )
        if normalize:
            arr /= 255.0
        return arr

    def resize(self, new_width, new_height):
        """Resize image to (new_width, new_height)."""
        img = PILImage.open(self.filepath).convert("RGB")
        img = img.resize((new_width, new_height))
        self.width, self.height = new_width, new_height
        img_data = np.array(img)
        self.pixels = [
            [Pixel(r, g, b) for (r, g, b) in row]
            for row in img_data
        ]

    def show(self):
        """Display the image."""
        img = PILImage.fromarray(self.to_numpy().astype(np.uint8))
        img.show()
