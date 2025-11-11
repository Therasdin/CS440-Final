class Pixel:
    def __init__(self, r=0, g=0, b=0):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)

    def __repr__(self):
        return f"Pixel(R={self.r}, G={self.g}, B={self.b})"

    def to_tuple(self):
        return (self.r, self.g, self.b)

    def normalize(self):
        return (self.r / 255.0, self.g / 255.0, self.b / 255.0)

    def brightness(self):
        return 0.299 * self.r + 0.587 * self.g + 0.114 * self.b  #https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
