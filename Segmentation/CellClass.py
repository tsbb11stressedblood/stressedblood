class CellClass:
    def __init__(self, _shapeData):
        # Should be set unknown at start
        self.label = "unknown"
        # Store data
        __data = _shapeData
        # Given from segmentation, RBC->ellipse (stuff[1][0], stuff[1][1])
        if (_shapeData[1][0] > _shapeData[1][1]):
            self.majorAxis = _shapeData[1][0]
            self.minorAxis = _shapeData[1][1]

        else:
            self.majorAxis = _shapeData[1][1]
            self.minorAxis = _shapeData[1][0]

        if self.minorAxis/self.majorAxis < 0.7:
                self.label = "RBC"


