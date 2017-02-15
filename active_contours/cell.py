class Cell:
    def __init__(self, _image_name, _contour):
        # Should be set unknown at start
        self.label = "unknown"

        # Store data
        self.img = _image_name
        self.nucleus_mask = "none"
        self.contour = _contour