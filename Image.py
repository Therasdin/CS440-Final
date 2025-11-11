from PIL import Image as PILImage
import numpy as np
import os

class Pixel:
    def __init__(self, r, g, b):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)

    def brightness(self):
        """Return average brightness (0–255)."""
        return (self.r + self.g + self.b) / 3

    def as_tuple(self):
        """Return pixel as an (R, G, B) tuple."""
        return (self.r, self.g, self.b)

    def __repr__(self):
        return f"Pixel(R={self.r}, G={self.g}, B={self.b})"


class Image:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.extension = os.path.splitext(file_path)[1].lower()

        # Load and normalize color mode
        self.image = PILImage.open(file_path)
        if self.image.mode != "RGB":
            self.image = self.image.convert("RGB")

        # Convert to NumPy array
        self.pixels_np = np.array(self.image)

        # Extract dimensions
        self.height, self.width, _ = self.pixels_np.shape

        # Create 2D list of Pixel objects for easy per-pixel access
        self.pixels = [
            [Pixel(*self.pixels_np[y, x]) for x in range(self.width)]
            for y in range(self.height)
        ]

    def get_pixel(self, x, y):
        """Return the Pixel object at (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pixels[y][x]
        else:
            raise IndexError("Pixel coordinates out of range.")

    def average_brightness(self):
        """Return average brightness across all pixels."""
        total_brightness = sum(
            self.pixels[y][x].brightness() for y in range(self.height) for x in range(self.width)
        )
        return total_brightness / (self.width * self.height)

    def resize(self, new_width, new_height):
        """Resize the image and regenerate pixels."""
        self.image = self.image.resize((new_width, new_height))
        self.pixels_np = np.array(self.image)
        self.width, self.height = new_width, new_height
        self.pixels = [
            [Pixel(*self.pixels_np[y, x]) for x in range(self.width)]
            for y in range(self.height)
        ]

    def to_numpy(self):
        """Return NumPy array version (for CNN input)."""
        return self.pixels_np

    def __repr__(self):
        return f"Image(filename='{self.filename}', size={self.width}x{self.height}, mode='RGB')"
