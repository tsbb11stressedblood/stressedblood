class Cell:
    def __init__(self, _image, _contour, _mask):
        # Should be set unknown at start
        self.label = "unknown"

        # Store data
        self.img = _image
        self.nucleus_mask = "none"
        self.contour = _contour
        self.mask = _mask