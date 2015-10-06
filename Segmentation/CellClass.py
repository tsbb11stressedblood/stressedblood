""" Cell_class: The initial cell classifier which sets unknown labels and stores data for each segmented cell.
The first check is to calculate the inertia ratio of the cell, a ellipse is probably a RBC.

Author: Marcus Fallqvist and Abdai Ahmed
Date: 2015-10-06
"""

class Cell:
    def __init__(self, _shape_data, _x, _y, _w, _h):
        # Should be set unknown at start
        self.label = "unknown"
        # Store data
        x = _x
        y = _y
        w = _w
        h = _h
        __data = _shape_data
        # Given from segmentation, RBC->ellipse (stuff[1][0], stuff[1][1])
        if (_shape_data[1][0] > _shape_data[1][1]):
            self.major_axis = _shape_data[1][0]
            self.minor_axis = _shape_data[1][1]

        else:
            self.major_axis = _shape_data[1][1]
            self.minor_axis = _shape_data[1][0]

        if self.minor_axis/self.major_axis < 0.7:
                self.label = "RBC"


