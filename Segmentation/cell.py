""" Cell_class: The initial cell classifier which sets unknown labels and stores data for each segmented cell.
The first check is to calculate the inertia ratio of the cell, a ellipse is probably a RBC.

Author: Marcus Fallqvist and Abdai Ahmed
Date: 2015-10-06
"""

class Cell:
    def __init__(self, _shape_data, _x, _y, _w, _h, _area,_marker, cell_img, cell_list):
        # Should be set unknown at start
        self.label = "unknown"

        self.img = cell_img

        # Store data
        self.marker = _marker
        self.x = _x
        self.y = _y
        self.w = _w
        self.h = _h
        self._data = _shape_data
        self.area = _area
        # Given from segmentation, RBC->ellipse (stuff[1][0], stuff[1][1])
        if (_shape_data[1][0] > _shape_data[1][1]):
            self.major_axis = _shape_data[1][0]
            self.minor_axis = _shape_data[1][1]

        else:
            self.major_axis = _shape_data[1][1]
            self.minor_axis = _shape_data[1][0]


"""  # Get the mean RBC size
        RBC_mean_area = 0
        RBC_counter = 0
        for object in cell_list:
            if object.label == "RBC":
                RBC_mean_area += object.area
                RBC_counter += 1
        if RBC_counter == 0:
        # No RBC in the list yet
            RBC_mean_area = area
        else: # Take the mean size of all RBC in the list
            RBC_mean_area = RBC_mean_area/RBC_counter

        #print self.area
        #print RBC_mean_area
        # Check if its an ellipse shape and is proper size
        if self.minor_axis/self.major_axis < 0.7:
            if 0.6*RBC_mean_area < self.area < 1.4*RBC_mean_area:
                self.label = "RBC"
            else:
                self.label = "Background"
        else:
            self.label = "Unknown"
"""

